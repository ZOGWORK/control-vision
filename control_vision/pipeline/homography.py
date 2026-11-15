from __future__ import annotations

import math

import cv2
import numpy as np


FIELD_FEET = 12.0
CORNER_ORDER = "top-left, top-right, bottom-right, bottom-left"
# A corner whose two edges are within about 2 degrees of a straight line makes
# the perspective fit unstable, so treat it as three collinear points.
_MIN_CORNER_SINE = math.sin(math.radians(2.0))


def _validate_corners(source: np.ndarray) -> None:
    if source.shape != (4, 2) or not np.all(np.isfinite(source)):
        raise ValueError(f"Select four field corners in order: {CORNER_ORDER}.")
    signs = []
    for index in range(4):
        previous_point, point, next_point = source[index - 1], source[index], source[(index + 1) % 4]
        incoming = point - previous_point
        outgoing = next_point - point
        lengths = float(np.linalg.norm(incoming)) * float(np.linalg.norm(outgoing))
        if lengths < 1e-6:
            raise ValueError("Two of the selected field corners are at the same spot. Select four separate corners.")
        cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        if abs(cross) / lengths < _MIN_CORNER_SINE:
            raise ValueError("Three of the selected field corners are in a straight line. Select the four corners of the field.")
        signs.append(cross > 0)
    if not all(signs) and any(signs):
        raise ValueError("The selected field corners cross over or fold inward instead of outlining the field. "
                         f"Select them in order: {CORNER_ORDER}.")
    # Image y grows downward, so top-left, top-right, bottom-right, bottom-left
    # turns clockwise on screen, which is a positive cross product here.
    if not all(signs):
        raise ValueError(f"The selected field corners are in reverse order. Select them in order: {CORNER_ORDER}.")
    if abs(cv2.contourArea(source)) < 100:
        raise ValueError("The selected field corners are too close together.")


def matrix_from_corners(corners: list[list[float]]) -> np.ndarray:
    if len(corners) != 4:
        raise ValueError(f"Select four field corners in order: {CORNER_ORDER}.")
    try:
        source = np.asarray(corners, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Select four field corners in order: {CORNER_ORDER}.") from exc
    _validate_corners(source)
    target = np.asarray([[0, 0], [FIELD_FEET, 0], [FIELD_FEET, FIELD_FEET], [0, FIELD_FEET]], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target)


def project_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = np.asarray([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, matrix)[0, 0]
    return float(projected[0]), float(projected[1])


def reprojection_error_px(corners: list[list[float]]) -> float:
    """Return the RMS residual of the four calibration corners in source pixels.

    A homography has eight degrees of freedom, so four corners are fitted
    exactly and this value is always near zero. It only confirms the fit is
    numerically sound; it does not measure calibration accuracy and cannot
    show whether the brief's < 5 px reprojection target is met. That needs
    held-out check points (field landmarks not used for the fit) compared
    against their projected positions.
    """
    matrix = matrix_from_corners(corners)
    target = np.asarray([[0, 0], [FIELD_FEET, 0], [FIELD_FEET, FIELD_FEET], [0, FIELD_FEET]], dtype=np.float32)
    projected = cv2.perspectiveTransform(target.reshape(1, 4, 2), np.linalg.inv(matrix))[0]
    source = np.asarray(corners, dtype=np.float32)
    return float(np.sqrt(np.mean(np.sum((projected - source) ** 2, axis=1))))
