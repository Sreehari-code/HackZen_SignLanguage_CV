"""Small, explicit temporal detector for a deliberate open-palm wave."""
from __future__ import annotations

from collections import deque


class WaveDetector:
    def __init__(self, settings: dict):
        self.settings = settings
        self.samples: deque[tuple[float, float]] = deque()
        self._smooth_x: float | None = None
        self._extreme: float | None = None
        self._direction = 0
        self._reversals: list[float] = []
        self._last_x: float | None = None
        self._last_time: float | None = None
        self._last_motion_time: float | None = None
        self._last_event = float("-inf")

    def reset(self) -> None:
        self.samples.clear()
        self._smooth_x = self._extreme = self._last_x = self._last_time = self._last_motion_time = None
        self._direction = 0
        self._reversals.clear()

    def recent_motion(self, now: float, seconds: float = .45) -> bool:
        return self._last_motion_time is not None and now - self._last_motion_time <= seconds

    def update(self, category: str | None, landmarks, now: float) -> bool:
        """Signal only after an open-palm lateral wave and a short finish pause."""
        if category != "Open_Palm" or len(landmarks or ()) != 21:
            self.reset()
            return False
        if now - self._last_event < float(self.settings["cooldown_seconds"]):
            return False
        wrist = landmarks[0]
        palm_width = ((landmarks[5].x-landmarks[17].x)**2 +
                      (landmarks[5].y-landmarks[17].y)**2) ** .5
        if palm_width < 1e-4:
            self.reset()
            return False
        x = float(wrist.x / palm_width)
        alpha = .32
        self._smooth_x = x if self._smooth_x is None else alpha*x + (1-alpha)*self._smooth_x
        x = self._smooth_x
        window = float(self.settings["window_seconds"])
        while self.samples and now - self.samples[0][0] > window:
            self.samples.popleft()
        self.samples.append((now, x))
        amplitude = float(self.settings["minimum_reversal_amplitude_palm_widths"])
        if self._extreme is None:
            self._extreme, self._last_x, self._last_time, self._last_motion_time = x, x, now, now
            return False
        delta = x - self._extreme
        direction = 1 if delta >= amplitude else -1 if delta <= -amplitude else 0
        if direction:
            if self._direction and direction != self._direction:
                self._reversals.append(now)
            self._direction = direction
            self._extreme = x
            self._reversals = [t for t in self._reversals if now - t <= window]
            self._last_motion_time = now
        elif self._last_x is not None and abs(x - self._last_x) > amplitude * .15:
            self._last_motion_time = now
        self._last_x, self._last_time = x, now
        if len(self._reversals) < int(self.settings["minimum_reversals"]):
            return False
        drift = abs(self.samples[-1][1] - self.samples[0][1])
        if drift > float(self.settings["maximum_net_drift_palm_widths"]):
            return False
        if self._last_motion_time is not None and now - self._last_motion_time >= float(self.settings["finish_pause_seconds"]):
            self._last_event = now
            self.reset()
            return True
        return False
