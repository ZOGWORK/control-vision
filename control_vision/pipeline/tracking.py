from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import cv2

from control_vision.pipeline.homography import matrix_from_corners, project_point


def track_video(
    video_path: Path,
    model_path: Path,
    corners: list[list[float]],
    seed_points: list[tuple[int, int, int]],
    sample_fps: float,
    cancel: Callable[[], bool],
    progress: Callable[[str, float, str], None],
    logger: logging.Logger,
) -> list[tuple[int, int, float, float, float, float]]:
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
    sampled = 0
    frame_index = 0
    empty = 0
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
                result = model.track(frame, persist=True, tracker=str(tracker_cfg), imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
            except Exception:
                logger.exception("Detection failed on frame %d; skipped", frame_index)
                frame_index += 1
                sampled += 1
                progress("Detect and track", min(100.0, 100 * sampled / sampled_total), f"Skipped unreadable frame at {t_s:.1f}s")
                continue
            boxes = result.boxes
            detections = []
            if boxes is not None and len(boxes):
                for box in boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    tid = int(box.id[0]) if box.id is not None else -1
                    conf = float(box.conf[0])
                    detections.append((tid, (x1 + x2) / 2, y2, conf))
            if sampled < 30 and not detections:
                empty += 1
            if sampled == 29 and empty == 30:
                raise RuntimeError("No robots were detected in the first 30 sampled frames. Check the video and robot model.")
            if not identities_seeded and detections:
                available = {index for index, robot_id in enumerate(robot_ids) if robot_id not in seed_to_track}
                assigned_track_ids = set(seed_to_track.values())
                for tid, x, y, _ in detections:
                    if not available or tid < 0 or tid in assigned_track_ids:
                        continue
                    index = min(available, key=lambda i: (seed_points[i][1] - x) ** 2 + (seed_points[i][2] - y) ** 2)
                    seed_to_track[robot_ids[index]] = tid
                    available.remove(index)
                    assigned_track_ids.add(tid)
                identities_seeded = len(seed_to_track) == len(robot_ids)
                logger.info("Seeded %d of %d identities from first usable sampled frame", len(seed_to_track), len(robot_ids))
            for tid, x, y, conf in detections:
                robot_id = next((rid for rid, assigned in seed_to_track.items() if assigned == tid), None)
                if robot_id is None:
                    continue
                x_ft, y_ft = project_point(matrix, x, y)
                rows.append((robot_id, frame_index, t_s, x_ft, y_ft, conf))
            sampled += 1
            progress("Detect and track", min(100.0, 100 * sampled / sampled_total), f"Processed {sampled:,} sampled frames")
            frame_index += 1
    finally:
        cap.release()
    logger.info("Sampled frames: %d; mapped track points: %d", sampled, len(rows))
    return rows
