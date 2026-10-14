"""Train and evaluate the optional custom landmark Random Forest model."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.custom_classifier import FEATURE_COUNT, REJECT_LABEL

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "gesture_samples.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "custom_gesture_model.joblib")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    columns = [f"f{i}" for i in range(FEATURE_COUNT)]
    expected = ["label", *columns]
    if not args.input.is_file():
        parser.error(f"Samples file not found: {args.input}. Run collect_gesture_data.py first.")
    features, labels = [], []
    with args.input.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected:
            parser.error("CSV columns are invalid; expected label,f0,...,f62. Recollect samples with this project version.")
        allowed = {key for key, item in config.custom_gestures.items() if item.get("enabled", True)} | {REJECT_LABEL}
        for row_number, row in enumerate(reader, start=2):
            label = row["label"]
            if label not in allowed:
                parser.error(f"CSV row {row_number} has unknown or disabled label {label!r}.")
            try:
                vector = [float(row[name]) for name in columns]
            except (TypeError, ValueError):
                parser.error(f"CSV row {row_number} contains a non-numeric feature.")
            if not np.isfinite(vector).all():
                parser.error(f"CSV row {row_number} contains a non-finite feature.")
            features.append(vector)
            labels.append(label)
    counts = Counter(labels)
    if len(set(counts) - {REJECT_LABEL}) < 2:
        parser.error("Collect samples for at least two custom phrase classes before training.")
    if REJECT_LABEL not in counts:
        parser.error("Collect at least 20 __OTHER__ rejection samples, including built-in gestures, to train safe fallback behavior.")
    if any(count < 20 for count in counts.values()):
        parser.error("Need at least 20 samples for every class; 100+ diverse samples per class is recommended.")
    if len(labels) < 40:
        parser.error("Too few samples for a training/validation split.")

    x_train, x_valid, y_train, y_valid = train_test_split(
        np.asarray(features), np.asarray(labels), test_size=.25, random_state=42, stratify=labels)
    classifier = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42,
                                        n_jobs=-1, min_samples_leaf=2)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_valid)
    classes = [str(value) for value in classifier.classes_]
    print(f"Samples: {len(labels)}; classes: {len(counts)}; split: {len(y_train)} train / {len(y_valid)} validation")
    print("Counts:", dict(counts))
    print("Validation report:\n", classification_report(y_valid, predictions, labels=classes, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; class order below):\n", confusion_matrix(y_valid, predictions, labels=classes))
    print("Class order:", classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format": "signbridge-rf-v1", "feature_count": FEATURE_COUNT,
                 "classes": classes, "sample_counts": dict(counts),
                 "validation_accuracy": float(classifier.score(x_valid, y_valid)),
                 "classifier": classifier}, args.output)
    print(f"Saved model: {args.output}")
    print("Review per-class recall and the confusion matrix; a saved model is not proof of reliable real-world recognition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
