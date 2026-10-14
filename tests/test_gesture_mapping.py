import json
import tempfile
import unittest
from pathlib import Path

from src.config import load_config
from src.sentence_manager import SentenceManager


class GestureMappingTests(unittest.TestCase):
    def test_loads_configuration_and_mapping(self):
        config = load_config(Path(__file__).parents[1] / "config.json")
        self.assertEqual(config.gesture_phrases["Open_Palm"], "HELLO")
        self.assertTrue(config.speak_each_phrase)
        self.assertTrue(config.model_path.name.endswith(".task"))

    def test_invalid_threshold_is_rejected(self):
        source = json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8"))
        source["confidence_thresholds"]["builtin"] = 1.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "confidence_threshold"):
                load_config(path)

    def test_custom_and_motion_phrase_definitions_are_unique(self):
        config = load_config(Path(__file__).parents[1] / "config.json")
        self.assertEqual(config.custom_gestures["PLEASE"]["phrase"], "PLEASE")
        self.assertEqual(config.motion_gestures["GOODBYE_WAVE"]["phrase"], "GOODBYE")

    def test_every_requested_output_is_configured_once(self):
        config = load_config(Path(__file__).parents[1] / "config.json")
        phrases = [item["phrase"] for group in (config.builtin_gestures, config.custom_gestures,
                                                   config.motion_gestures) for item in group.values()]
        expected = {
            "HELLO", "GOODBYE", "YES", "NO", "PLEASE", "THANK YOU", "SORRY", "HELP", "STOP",
            "WAIT", "FINISHED", "I HAVE A QUESTION", "I DON'T UNDERSTAND", "REPEAT PLEASE",
            "LEFT", "RIGHT", "UP", "DOWN", "NEXT", "GO BACK", "CONFIRM", "CANCEL", "SELECT",
            "SMALL", "THREE", "FOUR", "GOOD LUCK", "RELAX", "OK", "PEACE", "I LOVE YOU",
            "WATER", "FOOD", "REST", "HAPPY", "SAD", "PAIN", "FRIEND", "FAMILY",
        }
        self.assertEqual(set(phrases), expected)
        self.assertEqual(len(phrases), len(expected))

    def test_duplicate_output_phrase_is_rejected(self):
        source = json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8"))
        source["custom_static_gestures"]["PLEASE"]["phrase"] = "HELLO"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique output phrases"):
                load_config(path)

    def test_invalid_auto_speech_setting_is_rejected(self):
        source = json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8"))
        source["speak_each_phrase"] = "yes"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "speak_each_phrase"):
                load_config(path)

    def test_mapping_phrase_is_used(self):
        manager = SentenceManager({"Open_Palm": "WELCOME"}, 0.4, 1, 1.0)
        self.assertEqual(manager.observe("Open_Palm", 0.8, 0), "WELCOME")


if __name__ == "__main__":
    unittest.main()
