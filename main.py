"""SignBridge: real-time hand gesture communication demo."""
from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from pathlib import Path

import cv2

from src.camera import open_camera
from src.config import load_config
from src.custom_classifier import CustomGestureClassifier, REJECT_LABEL
from src.features import normalized_features
from src.gesture_recognition import GestureRecognizer, ensure_model
from src.motion_recognition import WaveDetector
from src.sentence_manager import SentenceManager
from src.speech import SpeechWorker
from src.ui import render

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("signbridge")


def parse_args():
    parser = argparse.ArgumentParser(description="SignBridge gesture-to-phrase demo")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json", help="Path to configuration JSON")
    parser.add_argument("--check", action="store_true", help="Validate configuration and component imports without opening a camera")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if sys.version_info < (3, 11):
        LOG.error("Python 3.11 or newer is required; Python 3.11 or 3.13 is recommended.")
        return 2
    try:
        config = load_config(args.config)
    except ValueError as exc:
        LOG.error("%s", exc)
        return 2
    enabled_groups = (config.builtin_gestures, config.custom_gestures, config.motion_gestures)
    phrase_map = {key: item["phrase"] for group in enabled_groups for key, item in group.items()
                  if item.get("enabled", True)}
    manager = SentenceManager(phrase_map, config.builtin_threshold,
                              config.stable_frames, config.cooldown_seconds, config.max_history)
    if args.check:
        LOG.info("Configuration valid: %d built-in, %d custom-trainable, %d motion mappings; Python %s on %s",
                 len(config.builtin_gestures), len(config.custom_gestures), len(config.motion_gestures),
                 platform.python_version(), platform.system())
        return 0
    if platform.system() != "Windows":
        LOG.warning("This demo targets Windows; camera/audio behavior may differ on %s.", platform.system())
    if sys.version_info[:2] not in ((3, 11), (3, 13)):
        LOG.warning("Python 3.11 or 3.13 is recommended; running under %s.", platform.python_version())

    camera = recognizer = speech = classifier = None
    try:
        try:
            model = ensure_model(config.model_path, config.model_url)
            recognizer = GestureRecognizer(model)
        except Exception as exc:
            LOG.error("Gesture recognizer unavailable. Check the model file and internet connection: %s", exc)
            return 2
        allowed_custom = {key for key, item in config.custom_gestures.items() if item.get("enabled", True)}
        try:
            classifier = CustomGestureClassifier(config.custom_model_path, allowed_custom | {REJECT_LABEL}, config.custom_threshold)
        except Exception as exc:
            LOG.error("Custom model could not be loaded; continuing with built-in and motion modes: %s", exc)
            classifier = CustomGestureClassifier(config.custom_model_path.with_name("missing-custom-model.joblib"),
                                                 allowed_custom | {REJECT_LABEL}, config.custom_threshold)
        if classifier.available:
            LOG.info("Loaded locally trained custom classes: %s", ", ".join(classifier.classes))
        else:
            LOG.info("No custom model found at %s; running built-in gestures and GOODBYE wave only.",
                     config.custom_model_path)
            LOG.info("Collect labeled samples with collect_gesture_data.py, then create the model with train_gesture_model.py.")
        try:
            camera = open_camera(config.camera_index)
        except RuntimeError as exc:
            LOG.error("%s", exc)
            return 2
        speech = SpeechWorker()
        cv2.namedWindow("SignBridge | Real-Time Hand Gesture Communication", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SignBridge | Real-Time Hand Gesture Communication", 1120, 780)
        show_help, show_landmarks = True, True
        category, confidence, phrase = "", 0.0, ""
        recognition_status = "Recognition active"
        last_recognition_error = ""
        mode = "Built-in"
        wave_detector = WaveDetector(config.motion_settings)
        open_palm_started = None
        previous_frame_time = time.perf_counter()
        fps = 0.0
        while True:
            ok, frame = camera.read()
            if not ok:
                recognition_status = "Camera frame unavailable; check camera connection/permissions"
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                result = recognizer.recognize(rgb)
                now = time.perf_counter()
                hand = result.hand_landmarks[0] if result.hand_landmarks else None
                prediction = result.gestures[0][0] if result.gestures and result.gestures[0] else None
                builtin_category = prediction.category_name if prediction else ""
                builtin_confidence = float(prediction.score) if prediction else 0.0
                builtin_def = config.builtin_gestures.get(builtin_category)
                builtin_valid = bool(builtin_def and builtin_def.get("enabled", True) and
                                     builtin_confidence >= config.builtin_threshold)
                motion_definition = config.motion_gestures.get("GOODBYE_WAVE", {})
                wave_event = False
                if motion_definition.get("enabled", True) and hand is not None:
                    motion_start = builtin_category if builtin_confidence >= config.builtin_threshold else None
                    wave_event = wave_detector.update(motion_start, hand, now)
                else:
                    wave_detector.reset()
                open_palm_ready = False
                if builtin_category == "Open_Palm" and builtin_valid:
                    if open_palm_started is None:
                        open_palm_started = now
                    dwell = float(config.motion_settings["open_palm_static_dwell_seconds"])
                    open_palm_ready = (now - open_palm_started >= dwell and
                                       not wave_detector.recent_motion(now))
                else:
                    open_palm_started = None
                custom_id, custom_confidence = None, 0.0
                if hand is not None and classifier.available:
                    custom_id, custom_confidence = classifier.predict(normalized_features(hand))

                if custom_id is not None and custom_id in config.custom_gestures:
                    definition = config.custom_gestures[custom_id]
                    category, confidence, mode = custom_id, custom_confidence, "Custom static"
                    phrase = definition["phrase"]
                    accepted = manager.observe(custom_id, confidence, now, config.custom_threshold)
                    recognition_status = f"Stable {manager.stability[0]}/{manager.stability[1]} frames"
                elif builtin_valid and builtin_category == "Open_Palm" and not open_palm_ready and not wave_event:
                    category, confidence, mode = builtin_category, builtin_confidence, "Built-in"
                    phrase = builtin_def["phrase"]
                    manager.observe(None, 0.0, now)
                    recognition_status = "Hold palm still for HELLO or wave side-to-side for GOODBYE"
                    accepted = None
                elif builtin_valid:
                    category, confidence, mode = builtin_category, builtin_confidence, "Built-in"
                    phrase = builtin_def["phrase"]
                    accepted = manager.observe(builtin_category, confidence, now, config.builtin_threshold)
                    recognition_status = f"Stable {manager.stability[0]}/{manager.stability[1]} frames"
                else:
                    category, confidence, phrase = (builtin_category, builtin_confidence, "") if prediction else ("", 0.0, "")
                    manager.observe(None, 0.0, now)
                    if hand is None:
                        recognition_status = "Show one hand to the camera"
                        mode = "Built-in"
                    elif classifier.available:
                        recognition_status = "No supported match / low confidence; ignored"
                        mode = "Custom static"
                    else:
                        recognition_status = "Custom poses not trained; collect samples, then train model"
                        mode = "Built-in only"
                    accepted = None

                if accepted:
                    LOG.info("Added phrase: %s", accepted)
                    if config.speak_each_phrase:
                        speech.speak(accepted)

                if wave_event:
                    motion_phrase = motion_definition["phrase"]
                    accepted = manager.add_confirmed("GOODBYE_WAVE", motion_phrase, now)
                    if accepted:
                        category, confidence, phrase, mode = "Open-palm wave", 1.0, motion_phrase, "Motion"
                        recognition_status = "Wave complete"
                        LOG.info("Added motion phrase: %s", accepted)
                        if config.speak_each_phrase:
                            speech.speak(accepted)
                if show_landmarks:
                    recognizer.draw_landmarks(frame, result)
            except Exception as exc:
                category, confidence, phrase = "", 0.0, ""
                mode = "Built-in"
                recognition_status = f"Recognition error: {exc}"
                # Log the first failure only: a missing/changed API should not
                # flood the terminal once per camera frame.
                if recognition_status != last_recognition_error:
                    LOG.exception("Frame recognition failed")
                    last_recognition_error = recognition_status
            else:
                last_recognition_error = ""
            now = time.perf_counter()
            elapsed = now - previous_frame_time
            if elapsed > 0:
                instant = 1 / elapsed
                fps = instant if fps == 0 else .85 * fps + .15 * instant
            previous_frame_time = now
            view = render(frame, category=f"{mode} | {category}" if category else "", confidence=confidence, phrase=phrase,
                          sentence=manager.sentence, history=list(manager.history),
                          camera_status=f"CAMERA {config.camera_index} • CONNECTED",
                          recognition_status=recognition_status, speech_status=speech.status, fps=fps,
                          show_help=show_help, stability=manager.stability)
            cv2.imshow("SignBridge | Real-Time Hand Gesture Communication", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                manager.clear()
            elif key in (8, 127):
                manager.undo()
            elif key == ord("s"):
                speech.speak(manager.sentence)
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("l"):
                show_landmarks = not show_landmarks
    except cv2.error as exc:
        LOG.error("OpenCV window/camera error: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOG.info("Interrupted; shutting down.")
    finally:
        if camera is not None:
            camera.release()
        if recognizer is not None:
            recognizer.close()
        if speech is not None:
            speech.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
