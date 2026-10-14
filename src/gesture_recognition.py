"""MediaPipe Tasks Gesture Recognizer setup and prediction."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

LOGGER = logging.getLogger(__name__)

# MediaPipe's hand topology indices (21 landmarks) from the Tasks hand model.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def ensure_model(model_path: Path, model_url: str) -> Path:
    """Fetch only the configured HTTPS model URL when the model is absent."""
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".download")
    try:
        request = urllib.request.Request(model_url, headers={"User-Agent": "SignBridge/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as target:
            if response.geturl().split(":", 1)[0].lower() != "https":
                raise RuntimeError("Model download redirected away from HTTPS.")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 100 * 1024 * 1024:
                    raise RuntimeError("Model download exceeded the 100 MB safety limit.")
                target.write(chunk)
        if total == 0:
            raise RuntimeError("Downloaded model file was empty.")
        temporary.replace(model_path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download MediaPipe model: {exc}") from exc
    return model_path


class GestureRecognizer:
    def __init__(self, model_path: Path) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._vision = vision
        self._mp = mp
        self._timestamp_ms = 0

    def recognize(self, frame_rgb):
        self._timestamp_ms += 1
        # MediaPipe 1.1.0 exposes Image/ImageFormat at the package root,
        # whereas older Tasks releases re-exported them differently.
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        return self._recognizer.recognize_for_video(image, self._timestamp_ms * 33)

    def draw_landmarks(self, frame, result) -> None:
        if not result.hand_landmarks:
            return
        height, width = frame.shape[:2]
        for hand in result.hand_landmarks:
            points = [(int(point.x * width), int(point.y * height)) for point in hand]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (38, 205, 190), 2, cv2.LINE_AA)
            for x, y in points:
                cv2.circle(frame, (x, y), 4, (245, 250, 255), -1, cv2.LINE_AA)

    def close(self) -> None:
        self._recognizer.close()
