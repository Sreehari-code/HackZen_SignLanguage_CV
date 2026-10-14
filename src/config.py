"""Application configuration loading and validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    camera_index: int
    builtin_threshold: float
    custom_threshold: float
    stable_frames: int
    cooldown_seconds: float
    speak_each_phrase: bool
    max_history: int
    model_path: Path
    model_url: str
    custom_model_path: Path
    builtin_gestures: dict[str, dict]
    custom_gestures: dict[str, dict]
    motion_gestures: dict[str, dict]
    motion_settings: dict

    @property
    def gesture_phrases(self) -> dict[str, str]:
        """Compatibility view of the configured built-in category mappings."""
        return {key: value["phrase"] for key, value in self.builtin_gestures.items() if value["enabled"]}


def load_config(path: str | Path) -> AppConfig:
    """Load JSON configuration and reject malformed/unsafe parameter values."""
    config_path = Path(path)
    try:
        data: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load configuration '{config_path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object.")
    builtin = data.get("builtin_gestures")
    custom = data.get("custom_static_gestures")
    motions = data.get("motion_gestures")
    if not isinstance(builtin, dict) or not builtin:
        raise ValueError("builtin_gestures must be a non-empty object.")
    if not isinstance(custom, dict) or not isinstance(motions, dict):
        raise ValueError("custom_static_gestures and motion_gestures must be objects.")
    for group_name, definitions in (("builtin_gestures", builtin), ("custom_static_gestures", custom), ("motion_gestures", motions)):
        for key, definition in definitions.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(definition, dict):
                raise ValueError(f"Every {group_name} entry needs an identifier and object definition.")
            if not isinstance(definition.get("phrase"), str) or not definition["phrase"].strip():
                raise ValueError(f"{group_name}.{key} needs a non-empty phrase.")
            if not isinstance(definition.get("enabled"), bool):
                raise ValueError(f"{group_name}.{key}.enabled must be explicitly true or false.")
            if group_name == "custom_static_gestures" and not isinstance(definition.get("pose_hint"), str):
                raise ValueError(f"custom_static_gestures.{key} needs a pose_hint for sample collection.")
            if group_name == "motion_gestures" and definition.get("start_builtin_category") not in builtin:
                raise ValueError(f"motion_gestures.{key} must reference a configured built-in starting pose.")
    identifiers = [key for group in (builtin, custom, motions) for key in group]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Gesture identifiers must be unique across recognition modes.")
    thresholds = data.get("confidence_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("confidence_thresholds must be an object with builtin and custom values.")
    threshold = thresholds.get("builtin")
    custom_threshold = thresholds.get("custom")
    stable_frames = data.get("stability_frames", 5)
    cooldown = data.get("cooldown_seconds", 1.25)
    speak_each_phrase = data.get("speak_each_phrase", True)
    camera = data.get("camera_index", 0)
    history = data.get("max_history", 12)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("confidence_thresholds.builtin must be between 0 and 1.")
    if isinstance(custom_threshold, bool) or not isinstance(custom_threshold, (int, float)) or not 0 <= custom_threshold <= 1:
        raise ValueError("confidence_thresholds.custom must be between 0 and 1.")
    if isinstance(stable_frames, bool) or not isinstance(stable_frames, int) or stable_frames < 1:
        raise ValueError("stable_frames must be a positive integer.")
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) or not 0 <= cooldown < float("inf"):
        raise ValueError("cooldown_seconds must be zero or greater.")
    if not isinstance(speak_each_phrase, bool):
        raise ValueError("speak_each_phrase must be true or false.")
    if isinstance(camera, bool) or not isinstance(camera, int) or camera < 0:
        raise ValueError("camera_index must be a non-negative integer.")
    if isinstance(history, bool) or not isinstance(history, int) or history < 1:
        raise ValueError("max_history must be a positive integer.")
    url = data.get("model_url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("model_url must be an HTTPS URL.")
    duplicate_phrases = [definition["phrase"] for group in (builtin, custom, motions)
                         for definition in group.values() if definition.get("enabled", True)]
    if len(duplicate_phrases) != len(set(duplicate_phrases)):
        raise ValueError("Enabled gesture definitions must have unique output phrases.")
    motion_settings = data.get("motion")
    if not isinstance(motion_settings, dict):
        raise ValueError("motion settings must be an object.")
    required_motion = ("open_palm_static_dwell_seconds", "window_seconds", "minimum_reversal_amplitude_palm_widths", "minimum_reversals",
                       "finish_pause_seconds", "cooldown_seconds", "maximum_net_drift_palm_widths")
    if any(key not in motion_settings for key in required_motion):
        raise ValueError("motion settings are missing one or more required thresholds.")
    if any(isinstance(motion_settings[key], bool) or not isinstance(motion_settings[key], (int, float)) or motion_settings[key] <= 0 for key in required_motion if key != "minimum_reversals"):
        raise ValueError("motion thresholds must be positive numbers.")
    if isinstance(motion_settings["minimum_reversals"], bool) or not isinstance(motion_settings["minimum_reversals"], int) or motion_settings["minimum_reversals"] < 2:
        raise ValueError("motion.minimum_reversals must be an integer of at least 2.")
    raw_model_path = data.get("model_path", "models/gesture_recognizer.task")
    if not isinstance(raw_model_path, str) or not raw_model_path.strip():
        raise ValueError("model_path must be a non-empty path string.")
    custom_raw = data.get("custom_model_path", "models/custom_gesture_model.joblib")
    if not isinstance(custom_raw, str) or not custom_raw.strip():
        raise ValueError("custom_model_path must be a non-empty path string.")
    return AppConfig(camera, float(threshold), float(custom_threshold), stable_frames, float(cooldown),
                     speak_each_phrase, history,
                     (config_path.parent / raw_model_path).resolve(), url,
                     (config_path.parent / custom_raw).resolve(), builtin, custom, motions, motion_settings)
