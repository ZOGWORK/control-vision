from __future__ import annotations

import cv2
import numpy as np


FIELD_FEET = 12.0


def matrix_from_corners(corners: list[list[float]]) -> np.ndarray:
    if len(corners) != 4:
        raise ValueError("Select four field corners in order: top-left, top-right, bottom-right, bottom-left.")
    source = np.asarray(corners, dtype=np.float32)
    if abs(cv2.contourArea(source)) < 100:
        raise ValueError("The selected field corners are too close together.")
    target = np.asarray([[0, 0], [FIELD_FEET, 0], [FIELD_FEET, FIELD_FEET], [0, FIELD_FEET]], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target)


def project_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = np.asarray([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, matrix)[0, 0]
    return float(projected[0]), float(projected[1])


def reprojection_error_px(corners: list[list[float]]) -> float:
    """Return the RMS corner residual in source pixels (near zero for four-point fit)."""
    matrix = matrix_from_corners(corners)
    target = np.asarray([[0, 0], [FIELD_FEET, 0], [FIELD_FEET, FIELD_FEET], [0, FIELD_FEET]], dtype=np.float32)
    projected = cv2.perspectiveTransform(target.reshape(1, 4, 2), np.linalg.inv(matrix))[0]
    source = np.asarray(corners, dtype=np.float32)
    return float(np.sqrt(np.mean(np.sum((projected - source) ** 2, axis=1))))
