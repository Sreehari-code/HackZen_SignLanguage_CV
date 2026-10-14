import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib

from src.config import load_config
from src.custom_classifier import CustomGestureClassifier
from src.features import normalized_features
from src.motion_recognition import WaveDetector


class FakeClassifier:
    classes_ = ["PLEASE"]

    def predict_proba(self, _rows):
        return [[.94]]


class FakeRejectClassifier:
    classes_ = ["__OTHER__"]

    def predict_proba(self, _rows):
        return [[.96]]


def fake_hand(wrist_x):
    points = [SimpleNamespace(x=wrist_x, y=.5, z=0.) for _ in range(21)]
    points[5] = SimpleNamespace(x=.6, y=.4, z=0.)
    points[17] = SimpleNamespace(x=.4, y=.4, z=0.)
    return points


class FeatureMotionTests(unittest.TestCase):
    def test_features_are_wrist_relative_and_scale_normalized(self):
        points = [SimpleNamespace(x=i / 100, y=i / 200, z=0.) for i in range(21)]
        values = normalized_features(points)
        self.assertEqual(len(values), 63)
        self.assertEqual(values[:3], [0., 0., 0.])
        scaled = [SimpleNamespace(x=p.x * 4, y=p.y * 4, z=p.z * 4) for p in points]
        self.assertAlmostEqual(values[3], normalized_features(scaled)[3])

    def test_wave_requires_multiple_direction_changes_and_finish_pause(self):
        config = load_config(Path(__file__).parents[1] / "config.json")
        detector = WaveDetector(config.motion_settings)
        x_positions = [0., 0., 1.2, 1.5, 1.5, .3, -1., -1.3, -1.3, -.2, 1., 1.3, 1.3, .1]
        events = []
        for index, x in enumerate(x_positions):
            now = index * .1
            if detector.update("Open_Palm", fake_hand(x), now):
                events.append(now)
        for index in range(20):
            now = (len(x_positions) + index) * .1
            if detector.update("Open_Palm", fake_hand(.1), now):
                events.append(now)
        self.assertEqual(len(events), 1)

    def test_wave_rejects_unstable_finger_pose(self):
        config = load_config(Path(__file__).parents[1] / "config.json")
        detector = WaveDetector(config.motion_settings)
        self.assertFalse(detector.update("Closed_Fist", fake_hand(0), 0.0))

    def test_missing_custom_model_falls_back_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier = CustomGestureClassifier(Path(directory) / "absent.joblib", {"PLEASE"}, .8)
            self.assertFalse(classifier.available)
            self.assertEqual(classifier.predict([0.] * 63), (None, 0.0))

    def test_local_custom_classifier_artifact_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump({"format": "signbridge-rf-v1", "feature_count": 63,
                         "classes": ["PLEASE"], "classifier": FakeClassifier()}, path)
            classifier = CustomGestureClassifier(path, {"PLEASE"}, .8)
            self.assertEqual(classifier.predict([0.] * 63), ("PLEASE", .94))

    def test_unknown_model_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump({"format": "signbridge-rf-v1", "feature_count": 63,
                         "classes": ["NOT_CONFIGURED"], "classifier": FakeClassifier()}, path)
            with self.assertRaisesRegex(ValueError, "not enabled"):
                CustomGestureClassifier(path, {"PLEASE"}, .8)

    def test_other_model_class_is_an_explicit_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump({"format": "signbridge-rf-v1", "feature_count": 63,
                         "classes": ["__OTHER__"], "classifier": FakeRejectClassifier()}, path)
            classifier = CustomGestureClassifier(path, {"PLEASE"}, .8)
            self.assertEqual(classifier.predict([0.] * 63), (None, .96))


if __name__ == "__main__":
    unittest.main()
