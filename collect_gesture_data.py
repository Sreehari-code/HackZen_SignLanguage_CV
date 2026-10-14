"""Collect labeled landmark rows for the optional custom pose classifier.

Only normalized 21-point landmarks and the selected label are written. Webcam
frames are used in memory for preview and are never saved to the CSV.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from src.camera import open_camera
from src.config import load_config
from src.custom_classifier import REJECT_LABEL
from src.features import normalized_features
from src.gesture_recognition import GestureRecognizer, ensure_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "gesture_samples.csv"
FIELDS = ["label"] + [f"f{i}" for i in range(63)]


def existing_counts(path: Path) -> Counter:
    counts = Counter()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                counts[row.get("label", "")] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="Configured custom class identifier (prompted if omitted)")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    labels = {key: value for key, value in config.custom_gestures.items() if value.get("enabled", True)}
    labels[REJECT_LABEL] = {"phrase": "(reject)", "pose_hint": "Collect common built-in poses and varied gestures that should not trigger a custom phrase"}
    label = args.label or input("Label to collect (type one configured class ID): ").strip()
    if label not in labels:
        print("Unknown/disabled label. Choose one of:")
        print(", ".join(labels))
        return 2
    print(f"Collecting {label} — {labels[label]['pose_hint']}")
    print("Use one consistent, distinct pose. Capture 100+ samples per class across different positions, users, and lighting.")
    print("Collect __OTHER__ examples of built-in categories (open palm, fist, thumbs, etc.) and non-command poses so the model can reject them.")
    model = ensure_model(config.model_path, config.model_url)
    recognizer = GestureRecognizer(model)
    camera = None
    try:
        camera = open_camera(config.camera_index)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        counts = existing_counts(args.output)
        print("Existing sample counts by label:", {name: counts[name] for name in labels})
        new_file = not args.output.exists() or args.output.stat().st_size == 0
        with args.output.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera frame could not be read.")
                    break
                frame = cv2.flip(frame, 1)
                result = recognizer.recognize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                recognizer.draw_landmarks(frame, result)
                hand = result.hand_landmarks[0] if result.hand_landmarks else None
                cv2.putText(frame, f"LABEL {label} | saved {counts[label]} (+ this session)", (18, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (50, 230, 210), 2, cv2.LINE_AA)
                cv2.putText(frame, "SPACE capture one landmark sample | Q/Esc finish", (18, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.imshow("SignBridge - Collect Gesture Samples", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if hand is None:
                        print("No hand detected; sample skipped.")
                        continue
                    features = normalized_features(hand)
                    writer.writerow({"label": label, **{f"f{i}": value for i, value in enumerate(features)}})
                    file.flush()
                    counts[label] += 1
                    print(f"Saved {label}: {counts[label]} total")
    finally:
        if camera is not None:
            camera.release()
        recognizer.close()
        cv2.destroyAllWindows()
    print(f"Samples saved to: {args.output}")
    print("Counts by label:", dict(existing_counts(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
