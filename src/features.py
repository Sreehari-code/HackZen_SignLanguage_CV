"""Scale-normalized feature extraction for 21 MediaPipe hand landmarks."""
from __future__ import annotations

import math


def normalized_features(landmarks) -> list[float]:
    """Return 63 wrist-relative, scale-normalized xyz values.

    Image axes and hand rotation are deliberately preserved so learned classes
    such as LEFT/RIGHT and UP/DOWN remain distinguishable. Capture varied poses
    and orientations for each user-defined class.
    """
    points = [(float(p.x), float(p.y), float(p.z)) for p in landmarks]
    if len(points) != 21:
        raise ValueError(f"Expected 21 hand landmarks, got {len(points)}")
    origin = points[0]
    relative = [(x-origin[0], y-origin[1], z-origin[2]) for x, y, z in points]
    scale = max(math.sqrt(x*x + y*y + z*z) for x, y, z in relative)
    if scale < 1e-8:
        raise ValueError("Hand landmarks have zero scale")
    return [coordinate / scale for point in relative for coordinate in point]
