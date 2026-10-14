"""Optional local Random Forest classifier for user-collected hand poses."""
from __future__ import annotations

from pathlib import Path

FEATURE_COUNT = 63
REJECT_LABEL = "__OTHER__"


class CustomGestureClassifier:
    def __init__(self, model_path: Path, allowed_classes: set[str], threshold: float):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.classes: tuple[str, ...] = ()
        if not model_path.is_file():
            return
        # joblib uses pickle internally. Only load this local artifact when it
        # has been produced by this project's training script or otherwise
        # explicitly trusted by the user. No remote models are fetched/executed.
        import joblib
        artifact = joblib.load(model_path)
        if not isinstance(artifact, dict) or artifact.get("format") != "signbridge-rf-v1":
            raise ValueError("Custom model metadata is invalid; train it with train_gesture_model.py.")
        if artifact.get("feature_count") != FEATURE_COUNT:
            raise ValueError("Custom model feature count does not match this SignBridge version.")
        classes = artifact.get("classes")
        permitted = allowed_classes | {REJECT_LABEL}
        if not isinstance(classes, list) or not classes or not set(classes).issubset(permitted):
            raise ValueError("Custom model contains classes not enabled in config.json.")
        self.model = artifact.get("classifier")
        self.classes = tuple(classes)
        if tuple(str(value) for value in getattr(self.model, "classes_", ())) != self.classes:
            raise ValueError("Custom model class metadata does not match its classifier output order.")

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, features: list[float]) -> tuple[str | None, float]:
        if self.model is None:
            return None, 0.0
        probabilities = self.model.predict_proba([features])[0]
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        score = float(probabilities[best])
        if score < self.threshold or self.classes[best] == REJECT_LABEL:
            return None, score
        return str(self.classes[best]), score
