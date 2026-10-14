import unittest

from src.sentence_manager import SentenceManager


class SentenceManagerTests(unittest.TestCase):
    def make_manager(self, stable=2, cooldown=1.0):
        return SentenceManager({"Open_Palm": "HELLO", "Victory": "PEACE"}, .7, stable, cooldown)

    def test_add_recognized_phrase_after_stable_frames(self):
        manager = self.make_manager()
        self.assertIsNone(manager.observe("Open_Palm", .9, 0))
        self.assertEqual(manager.observe("Open_Palm", .9, .1), "HELLO")
        self.assertEqual(manager.sentence, "HELLO")

    def test_unknown_gesture_is_ignored(self):
        manager = self.make_manager()
        self.assertIsNone(manager.observe("None", .99, 0))
        self.assertEqual(manager.entries, [])

    def test_suppresses_duplicate_until_cooldown_then_allows_again(self):
        manager = self.make_manager(stable=1)
        self.assertEqual(manager.observe("Open_Palm", .9, 0), "HELLO")
        self.assertIsNone(manager.observe("Open_Palm", .9, .5))
        self.assertEqual(manager.observe("Open_Palm", .9, 1.1), "HELLO")
        self.assertEqual(manager.entries, ["HELLO", "HELLO"])

    def test_different_gesture_can_be_added(self):
        manager = self.make_manager(stable=1)
        manager.observe("Open_Palm", .9, 0)
        self.assertEqual(manager.observe("Victory", .9, .1), "PEACE")

    def test_clear_and_undo(self):
        manager = self.make_manager(stable=1)
        manager.observe("Open_Palm", .9, 0)
        manager.observe("Victory", .9, .1)
        self.assertEqual(manager.undo(), "PEACE")
        self.assertEqual(manager.sentence, "HELLO")
        manager.clear()
        self.assertEqual(manager.sentence, "")
        self.assertIsNone(manager.undo())

    def test_confidence_threshold_resets_candidate(self):
        manager = self.make_manager(stable=2)
        manager.observe("Open_Palm", .9, 0)
        self.assertIsNone(manager.observe("Open_Palm", .2, .1))
        self.assertIsNone(manager.observe("Open_Palm", .9, .2))
        self.assertEqual(manager.observe("Open_Palm", .9, .3), "HELLO")


if __name__ == "__main__":
    unittest.main()
