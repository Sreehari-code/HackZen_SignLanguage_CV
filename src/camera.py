"""OpenCV camera initialization with helpful diagnostics."""
from __future__ import annotations

import cv2


def open_camera(index: int):
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Cannot open camera {index}. Check that it is connected, close other apps using it, "
            "and allow camera access for desktop apps in Windows Settings."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    return camera
