from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from control_vision.pipeline.homography import matrix_from_corners, project_point


def _alliance_mask(hsv: np.ndarray, alliance: str) -> np.ndarray:
    if alliance == "red":
        low = cv2.inRange(hsv, np.array((0, 75, 45)), np.array((12, 255, 255)))
        high = cv2.inRange(hsv, np.array((168, 75, 45)), np.array((180, 255, 255)))
        return cv2.bitwise_or(low, high)
    return cv2.inRange(hsv, np.array((88, 65, 35)), np.array((137, 255, 255)))


def _feature_points(gray: np.ndarray, anchor: tuple[float, float], width: int, height: int):
    x, y = anchor
    x0, y0 = max(0, round(x - width / 2)), max(0, round(y - height))
    x1, y1 = min(gray.shape[1], round(x + width / 2)), min(gray.shape[0], round(y + 3))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.012,
                                   minDistance=4, blockSize=7, mask=mask)


def _color_anchor(hsv: np.ndarray, alliance: str, anchor: tuple[float, float],
                  width: int, height: int):
    x, y = anchor
    x0, y0 = max(0, round(x - width * 0.85)), max(0, round(y - height * 1.15))
    x1, y1 = min(hsv.shape[1], round(x + width * 0.85)), min(hsv.shape[0], round(y + 10))
    if x1 - x0 < 24 or y1 - y0 < 20:
        return None
    crop = hsv[y0:y1, x0:x1]
    mask = _alliance_mask(crop, alliance)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 24:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 8 or bh < 4:
            continue
        candidate = (x0 + bx + bw / 2, y0 + by + bh)
        distance = ((candidate[0] - x) ** 2 + (candidate[1] - y) ** 2) ** 0.5
        if distance <= max(width * 0.9, height * 0.8):
            candidates.append((distance, area, candidate))
    if not candidates:
        return None
    distance, area, candidate = min(candidates, key=lambda item: (item[0], -item[1]))
    confidence = min(0.82, 0.52 + area / 1800.0) * max(0.25, 1.0 - distance / (width * 1.25))
    return candidate, confidence


def track_seeded_video(video_path: Path, corners: list[list[float]],
                       seed_points: list[tuple[int, int, int]], alliances: dict[int, str],
                       sample_fps: float, cancel: Callable[[], bool],
                       progress: Callable[[str, float, str], None], logger: logging.Logger,
                       photos_dir: Path | None = None,
                       patch_width: int = 112, patch_height: int = 88):
    """Track clicked robots with Lucas–Kanade, alliance-color reacquisition, and no model file.

    This is a deliberately local fallback for a human-initialized full-field video.
    The user still reviews track identity and point events; weak frames are omitted
    from the resulting tracks instead of being presented as confident positions.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read video file {video_path.name}.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame rate or frame count from {video_path.name}.")
    stride = max(1, round(fps / sample_fps))
    sampled_total = max(1, total_frames // stride)
    matrix = matrix_from_corners(corners)
    robot_ids = [int(item[0]) for item in seed_points]
    anchors = {int(rid): (float(x), float(y)) for rid, x, y in seed_points}
    seed_anchors = anchors.copy()
    features: dict[int, np.ndarray | None] = {}
    templates: dict[int, np.ndarray | None] = {}
    last_quality = {rid: 0.0 for rid in robot_ids}
    rows, best_photos = [], {}
    sampled = 0
    frame_index = 0
    prev_gray = None
    try:
        while frame_index < total_frames:
            if frame_index % stride:
                cap.grab()
                frame_index += 1
                continue
            if cancel():
                raise InterruptedError("Manual tracking was cancelled.")
            ok, frame = cap.read()
            if not ok:
                break
            t_s = frame_index / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for rid in robot_ids:
                x, y = anchors[rid]
                if prev_gray is None:
                    features[rid] = _feature_points(gray, anchors[rid], patch_width, patch_height)
                    x0, y0 = max(0, round(x - patch_width / 2)), max(0, round(y - patch_height))
                    x1, y1 = min(gray.shape[1], round(x + patch_width / 2)), min(gray.shape[0], round(y + 3))
                    templates[rid] = gray[y0:y1, x0:x1].copy() if x1 > x0 and y1 > y0 else None
                    last_quality[rid] = 1.0 if features[rid] is not None and len(features[rid]) >= 4 else 0.25
                else:
                    observed = None
                    quality = 0.0
                    old_points = features.get(rid)
                    if old_points is not None and len(old_points) >= 4:
                        new_points, status, errors = cv2.calcOpticalFlowPyrLK(
                            prev_gray, gray, old_points, None, winSize=(21, 21), maxLevel=3,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
                        if new_points is not None and status is not None:
                            good = status.ravel() == 1
                            if errors is not None:
                                good &= errors.ravel() < 35
                            src, dst = old_points.reshape(-1, 2)[good], new_points.reshape(-1, 2)[good]
                            if len(dst) >= 4:
                                affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                                             ransacReprojThreshold=4.0)
                                if affine is not None:
                                    dx, dy = float(affine[0, 2]), float(affine[1, 2])
                                    if abs(dx) < 100 and abs(dy) < 100:
                                        observed = (x + dx, y + dy)
                                        inlier_fraction = float(np.mean(inliers)) if inliers is not None else 0.5
                                        quality = min(0.78, 0.35 + 0.55 * inlier_fraction * min(1.0, len(dst) / 12))
                    # Optical flow can lock onto field texture after occlusion. Require
                    # visible alliance-color corroboration for a high-confidence point.
                    color = _color_anchor(hsv, alliances[rid], anchors[rid], patch_width, patch_height)
                    if color:
                        color_point, color_quality = color
                        if observed is None:
                            observed, quality = color_point, color_quality
                        elif np.hypot(observed[0] - color_point[0], observed[1] - color_point[1]) <= max(30, patch_width * 0.35):
                            observed = (0.4 * observed[0] + 0.6 * color_point[0],
                                        0.4 * observed[1] + 0.6 * color_point[1])
                            quality = min(0.9, 0.35 * quality + 0.65 * color_quality)
                        elif np.hypot(color_point[0] - x, color_point[1] - y) <= patch_width * 0.9:
                            observed, quality = color_point, color_quality
                    elif observed is not None:
                        # Preserve continuity for a short occlusion, but mark these
                        # points weak so they cannot masquerade as reviewed positions.
                        quality = min(quality, 0.28)
                    if observed is not None:
                        anchors[rid] = observed
                        last_quality[rid] = quality
                    else:
                        last_quality[rid] *= 0.75
                    features[rid] = _feature_points(gray, anchors[rid], patch_width, patch_height)
                    if features[rid] is None and last_quality[rid] < 0.15:
                        logger.info("Track %s weak/lost at %.2fs; omitted from this sampled frame", rid, t_s)
                        continue
                confidence = float(last_quality[rid])
                if confidence < 0.3:
                    continue
                x_ft, y_ft = project_point(matrix, *anchors[rid])
                if not (0 <= x_ft <= 12 and 0 <= y_ft <= 12):
                    # Bad calibration or a lost track must not enter the field map.
                    continue
                rows.append((rid, frame_index, t_s, x_ft, y_ft, confidence))
                crop_x, crop_y = anchors[rid]
                x0, y0 = max(0, round(crop_x - patch_width / 2)), max(0, round(crop_y - patch_height))
                x1, y1 = min(frame.shape[1], round(crop_x + patch_width / 2)), min(frame.shape[0], round(crop_y + 3))
                if x1 > x0 and y1 > y0:
                    quality_area = (x1 - x0) * (y1 - y0) * confidence
                    if rid not in best_photos or quality_area > best_photos[rid][0]:
                        best_photos[rid] = (quality_area, confidence, t_s, frame[y0:y1, x0:x1].copy())
            prev_gray = gray
            sampled += 1
            progress("Seeded optical-flow tracking", min(100.0, 100.0 * sampled / sampled_total),
                     f"Processed {sampled:,} of {sampled_total:,} sampled frames at {t_s:.1f}s")
            frame_index += 1
    finally:
        cap.release()
    photo_rows = _save_best_photos(best_photos, photos_dir, logger)
    logger.info("Manual tracker seed positions: %s; track points: %d; photos: %d", seed_anchors, len(rows), len(photo_rows))
    return rows, photo_rows, []


def _save_best_photos(best_photos, photos_dir, logger):
    photo_rows = []
    if photos_dir is None:
        return photo_rows
    photos_dir.mkdir(parents=True, exist_ok=True)
    for rid, (_, _, t_s, crop) in best_photos.items():
        if crop.shape[0] < 200 or crop.shape[1] < 200:
            logger.info("Robot %s preview photo skipped: best crop is %dx%d, below 200x200",
                        rid, crop.shape[1], crop.shape[0])
            continue
        path = photos_dir / f"robot-{rid}.png"
        if cv2.imwrite(str(path), crop):
            photo_rows.append((rid, path, t_s))
    return photo_rows
