"""Image file reads and writes that work on any Windows path.

cv2.imread and cv2.imwrite cannot open paths with non-ASCII characters on
Windows (a project folder named "Équipe" or in another script), and imwrite
reports that by returning False. Encoding and decoding in memory and using
Python's own file I/O avoids both problems.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def write_image(path: Path, image: np.ndarray, params: Sequence[int] = ()) -> bool:
    """Encode by the path's extension and write it; False if either step fails."""
    try:
        encoded, data = cv2.imencode(path.suffix or ".png", image, list(params))
    except cv2.error:
        return False
    if not encoded:
        return False
    try:
        path.write_bytes(data.tobytes())
    except OSError:
        path.unlink(missing_ok=True)
        return False
    return True


def read_image(path: Path) -> np.ndarray | None:
    """Decode a colour image, or None if it is missing or not an image."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
