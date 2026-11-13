from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import cv2

from control_vision.pipeline.homography import matrix_from_corners, project_point
from control_vision.pipeline.zones import point_in_polygon


def track_video(
    video_path: Path,
    model_path: Path,
    corners: list[list[float]],
    seed_points: list[tuple[int, int, int]],
    sample_fps: float,
    cancel: Callable[[], bool],
    progress: Callable[[str, float, str], None],
    logger: logging.Logger,
    photos_dir: Path | None = None,
    confidence_threshold: float = 0.25,
    goal_roi: list[list[float]] | None = None,
) -> tuple[list[tuple[int, int, float, float, float, float]], list[tuple[int, Path, float]], list[dict]]:
    """Run local YOLO tracking at the requested sampling rate and map robot footpoints.

    Each seed is (database robot id, seed x pixel, seed y pixel). Seeds are associated
    to the closest detector track in the first sample; afterward tracker IDs persist.
    """
    if not model_path.is_file():
        raise RuntimeError(f"YOLO11n model was not found at {model_path}. Add a trained local model in Settings.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is missing. Install the pinned dependencies from requirements.txt.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read video file {video_path.name}.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read the frame rate from {video_path.name}.")
    stride = max(1, round(fps / sample_fps))
    sampled_total = max(1, total_frames // stride)
    tracker_cfg = Path(__file__).resolve().parents[1] / "config" / "botsort.yaml"
    model = YOLO(str(model_path), task="detect")
    matrix = matrix_from_corners(corners)
    seed_to_track: dict[int, int] = {}
    identities_seeded = False
    robot_ids = [seed[0] for seed in seed_points]
    rows: list[tuple[int, int, float, float, float, float]] = []
    best_photos: dict[int, tuple[int, float, float, object]] = {}
    scoring_candidates: list[dict] = []
    sampled = 0
    frame_index = 0
    empty = 0
    stable_roi_count = 0
    pending_roi_count = 0
    pending_roi_since: float | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            if cancel():
                raise InterruptedError("Processing was cancelled.")
            t_s = frame_index / fps
            try:
                result = model.track(frame, persist=True, tracker=str(tracker_cfg), imgsz=640, conf=confidence_threshold, iou=0.5, verbose=False)[0]
            except Exception:
                logger.exception("Detection failed on frame %d; skipped", frame_index)
                frame_index += 1
                sampled += 1
                progress("Detect and track", min(100.0, 100 * sampled / sampled_total), f"Skipped unreadable frame at {t_s:.1f}s")
                continue
            boxes = result.boxes
            detections = []
            roi_balls = []
            robot_positions = []
            if boxes is not None and len(boxes):
                for box in boxes:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    if class_id == 0:
                        tid = int(box.id[0]) if box.id is not None else -1
                        detections.append((tid, center_x, y2, conf, (x1, y1, x2, y2)))
                    elif class_id in (1, 2) and goal_roi and point_in_polygon(center_x, center_y, goal_roi):
                        roi_balls.append((center_x, center_y, conf))
            if sampled < 30 and not detections:
                empty += 1
            if sampled == 29 and empty == 30:
                raise RuntimeError("No robots were detected in the first 30 sampled frames. Check the video and robot model.")
            logger.info("Sampled frame %d (%.3fs): %d robot detections; %d scoring elements in ROI", frame_index, t_s, len(detections), len(roi_balls))
            if not identities_seeded and detections:
                available = {index for index, robot_id in enumerate(robot_ids) if robot_id not in seed_to_track}
                assigned_track_ids = set(seed_to_track.values())
                for tid, x, y, _, _ in detections:
                    if not available or tid < 0 or tid in assigned_track_ids:
                        continue
                    index = min(available, key=lambda i: (seed_points[i][1] - x) ** 2 + (seed_points[i][2] - y) ** 2)
                    seed_to_track[robot_ids[index]] = tid
                    available.remove(index)
                    assigned_track_ids.add(tid)
                identities_seeded = len(seed_to_track) == len(robot_ids)
                logger.info("Seeded %d of %d identities from first usable sampled frame", len(seed_to_track), len(robot_ids))
            for tid, x, y, conf, bbox in detections:
                robot_id = next((rid for rid, assigned in seed_to_track.items() if assigned == tid), None)
                if robot_id is None:
                    continue
                x_ft, y_ft = project_point(matrix, x, y)
                rows.append((robot_id, frame_index, t_s, x_ft, y_ft, conf))
                robot_positions.append((robot_id, x, y))
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                margin_x, margin_y = width * 0.2, height * 0.2
                left, top = max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))
                right, bottom = min(frame.shape[1], int(x2 + margin_x)), min(frame.shape[0], int(y2 + margin_y))
                crop = frame[top:bottom, left:right].copy()
                area = width * height
                previous = best_photos.get(robot_id)
                if previous is None or (area, conf) > (previous[0], previous[1]):
                    best_photos[robot_id] = (area, conf, t_s, crop)
            current_roi_count = len(roi_balls)
            if goal_roi and current_roi_count > stable_roi_count:
                if current_roi_count != pending_roi_count:
                    pending_roi_count, pending_roi_since = current_roi_count, t_s
                elif pending_roi_since is not None and t_s - pending_roi_since >= 0.5:
                    for ball_x, ball_y, ball_conf in roi_balls[stable_roi_count:current_roi_count]:
                        if robot_positions:
                            robot_id = min(robot_positions, key=lambda pos: (pos[1] - ball_x) ** 2 + (pos[2] - ball_y) ** 2)[0]
                        else:
                            robot_id = robot_ids[0]
                        scoring_candidates.append({"robot_id": robot_id, "t_s": t_s, "points": 0,
                                                   "confidence": "high" if ball_conf >= 0.7 else "medium",
                                                   "status": "flagged", "source": "goal_roi", "event_type": "unclassified"})
                    stable_roi_count = current_roi_count
                    pending_roi_count, pending_roi_since = stable_roi_count, None
            elif goal_roi and current_roi_count < pending_roi_count:
                pending_roi_count = current_roi_count
                pending_roi_since = t_s if current_roi_count > stable_roi_count else None
            if goal_roi and current_roi_count < stable_roi_count:
                stable_roi_count = current_roi_count
                pending_roi_count, pending_roi_since = current_roi_count, None
            sampled += 1
            progress("Detect and track", min(100.0, 100 * sampled / sampled_total), f"Processed {sampled:,} sampled frames")
            frame_index += 1
    finally:
        cap.release()
    if sampled and sampled < 30 and empty == sampled:
        raise RuntimeError("No robots were detected in the first sampled frames. Check the video and robot model.")
    grouped: dict[int, list[tuple]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)
    rows = []
    for robot_rows in grouped.values():
        robot_rows.sort(key=lambda row: row[1])
        for index, row in enumerate(robot_rows):
            neighbors = robot_rows[max(0, index - 2):index + 3]
            rows.append((row[0], row[1], row[2], sum(r[3] for r in neighbors) / len(neighbors),
                         sum(r[4] for r in neighbors) / len(neighbors), row[5]))
    rows.sort(key=lambda row: (row[1], row[0]))
    logger.info("Sampled frames: %d; mapped track points: %d", sampled, len(rows))
    photo_rows = []
    if photos_dir:
        photos_dir.mkdir(parents=True, exist_ok=True)
        for robot_id, (_, _, t_s, crop) in best_photos.items():
            if crop.shape[0] < 200 or crop.shape[1] < 200:
                logger.info("Robot %s photo skipped: best crop is %dx%d, below 200x200", robot_id, crop.shape[1], crop.shape[0])
                continue
            path = photos_dir / f"robot-{robot_id}.png"
            if cv2.imwrite(str(path), crop):
                photo_rows.append((robot_id, path, t_s))
    return rows, photo_rows, scoring_candidates
