"""Phrase acceptance and sentence state, independent of camera/UI."""
from __future__ import annotations

import time
from collections import deque


class SentenceManager:
    def __init__(self, phrases: dict[str, str], threshold: float, stable_frames: int,
                 cooldown_seconds: float, max_history: int = 12) -> None:
        self.phrases = dict(phrases)
        self.threshold = threshold
        self.stable_frames = stable_frames
        self.cooldown_seconds = cooldown_seconds
        self.entries: list[str] = []
        self.history: deque[str] = deque(maxlen=max_history)
        self._candidate: str | None = None
        self._candidate_frames = 0
        self._last_accepted: str | None = None
        self._last_accepted_at = float("-inf")
        self._accepted_flash_until = float("-inf")

    @property
    def sentence(self) -> str:
        return " ".join(self.entries)

    @property
    def stability(self) -> tuple[int, int]:
        if time.monotonic() < self._accepted_flash_until:
            return self.stable_frames, self.stable_frames
        return self._candidate_frames, self.stable_frames

    def observe(self, category: str | None, confidence: float, now: float | None = None,
                threshold: float | None = None) -> str | None:
        """Return a phrase only on stable, confident recognition and cooldown expiry."""
        now = time.monotonic() if now is None else now
        phrase = self.phrases.get(category or "")
        if phrase is None or confidence < (self.threshold if threshold is None else threshold):
            self._candidate, self._candidate_frames = None, 0
            return None
        if phrase == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = phrase, 1
        if self._candidate_frames < self.stable_frames:
            return None
        # Repeated frames never append again. A different category resets this guard;
        # a held phrase becomes eligible again once its cooldown has elapsed.
        if phrase == self._last_accepted and now - self._last_accepted_at < self.cooldown_seconds:
            return None
        self.entries.append(phrase)
        self.history.append(phrase)
        self._last_accepted, self._last_accepted_at = phrase, now
        self._candidate_frames = 0
        self._accepted_flash_until = time.monotonic() + .7
        return phrase

    def add_confirmed(self, category: str, phrase: str, now: float | None = None) -> str | None:
        """Append an event already confirmed by an independent temporal rule."""
        now = time.monotonic() if now is None else now
        if not category or not phrase.strip():
            return None
        if phrase == self._last_accepted and now - self._last_accepted_at < self.cooldown_seconds:
            return None
        self.entries.append(phrase)
        self.history.append(phrase)
        self._last_accepted, self._last_accepted_at = phrase, now
        self._accepted_flash_until = time.monotonic() + .7
        self._candidate, self._candidate_frames = None, 0
        return phrase

    def clear(self) -> None:
        self.entries.clear()
        self.history.clear()
        self._reset_prediction()

    def undo(self) -> str | None:
        if not self.entries:
            return None
        value = self.entries.pop()
        if self.history and self.history[-1] == value:
            self.history.pop()
        self._reset_prediction()
        return value

    def _reset_prediction(self) -> None:
        self._candidate, self._candidate_frames = None, 0
        self._last_accepted = None
        self._last_accepted_at = float("-inf")
        self._accepted_flash_until = float("-inf")
