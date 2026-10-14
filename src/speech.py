"""Non-blocking, serialized offline text-to-speech."""
from __future__ import annotations

import logging
import queue
import threading

LOGGER = logging.getLogger(__name__)


class SpeechWorker:
    def __init__(self) -> None:
        # One worker serializes playback; a small queue absorbs quick gestures
        # without allowing speech requests to overlap or grow without bound.
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
        self._status = "Speech ready"
        self._thread = threading.Thread(target=self._run, name="SignBridgeSpeech", daemon=True)
        self._thread.start()

    @property
    def status(self) -> str:
        return self._status

    def speak(self, sentence: str) -> bool:
        sentence = sentence.strip()
        if not sentence:
            self._status = "Nothing to speak"
            return False
        try:
            self._queue.put_nowait(sentence)
            self._status = "Speaking sentence"
            return True
        except queue.Full:
            self._status = "Speech busy; wait for the current sentence"
            return False

    def _run(self) -> None:
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as exc:
            self._status = f"Speech unavailable: {exc}"
            LOGGER.warning("Could not initialize speech engine: %s", exc)
        while True:
            sentence = self._queue.get()
            if sentence is None:
                break
            if engine is None:
                self._status = "Speech unavailable; check pyttsx3/audio output"
                continue
            try:
                self._status = "Speaking sentence"
                engine.say(sentence)
                engine.runAndWait()
                self._status = "Speech ready"
            except Exception as exc:
                self._status = f"Speech error: {exc}"
                LOGGER.warning("Speech playback failed: %s", exc)
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def close(self) -> None:
        # Wait briefly for queued speech to finish, then stop the worker cleanly.
        try:
            self._queue.put(None, timeout=2)
        except queue.Full:
            return
        self._thread.join(timeout=2)
