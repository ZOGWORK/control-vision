from __future__ import annotations

import logging
import time
import tempfile
from pathlib import Path
from typing import Callable

import cv2

from control_vision.pipeline.homography import matrix_from_corners, project_point
from control_vision.pipeline.zones import point_in_polygon


def _new_roi_candidate(x: float, y: float, t_s: float,
                       recent: list[tuple[float, float, float]]) -> bool:
    """Avoid counting a briefly lost and reacquired goal element twice."""
    recent[:] = [item for item in recent if t_s - item[2] <= 2.0]
    return all((x - old_x) ** 2 + (y - old_y) ** 2 > 35 ** 2
               for old_x, old_y, _ in recent)


def _match_initial_seeds(seed_points, detections, assigned_robot_ids, assigned_track_ids, max_distance):
    """One-to-one seed assignment with a hard distance gate; never guess late entrants."""
    remaining = [seed for seed in seed_points if seed[0] not in assigned_robot_ids]
    candidates = [det for det in detections if det[0] >= 0 and det[0] not in assigned_track_ids]
    best: dict[int, tuple[int, float]] = {}

    def search(index: int, used: set[int], mapping: dict[int, tuple[int, float]], total_distance: float):
        nonlocal best
        if index == len(remaining):
            best_distance = sum(item[1] for item in best.values())
            if (len(mapping), -total_distance) > (len(best), -best_distance):
                best = dict(mapping)
            return
        robot_id, sx, sy = remaining[index]
        search(index + 1, used, mapping, total_distance)
        for detection in candidates:
            track_id, x, y = detection[:3]
            if track_id in used:
                continue
            distance = ((sx - x) ** 2 + (sy - y) ** 2) ** 0.5
            if distance <= max_distance:
                mapping[robot_id] = (track_id, distance)
                search(index + 1, used | {track_id}, mapping, total_distance + distance)
                del mapping[robot_id]

    search(0, set(), {}, 0.0)
    return best


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
    reid_model_path: str | None = None,
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
    tracker_override = None
    if reid_model_path:
        reid_path = Path(reid_model_path)
        if not reid_path.is_file():
            cap.release()
            raise RuntimeError(f"Configured ReID model was not found at {reid_path}.")
        try:
            import yaml
            config = yaml.safe_load(tracker_cfg.read_text(encoding="utf-8"))
            config["with_reid"] = True
            config["model"] = str(reid_path.resolve())
            tracker_override = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False)
            yaml.safe_dump(config, tracker_override)
            tracker_override.close()
            tracker_cfg = Path(tracker_override.name)
            logger.info("BoT-SORT appearance ReID enabled using local model: %s", reid_path)
        except Exception as exc:
            cap.release()
            raise RuntimeError(f"Could not configure the local ReID model: {exc}") from exc
    else:
        logger.warning("BoT-SORT appearance ReID is disabled; using motion and overlap association only.")
    model = YOLO(str(model_path), task="detect")
    matrix = matrix_from_corners(corners)
    seed_to_track: dict[int, int] = {}
    identities_seeded = False
    seed_assignment_deadline_s = 3.0
    robot_ids = [seed[0] for seed in seed_points]
    rows: list[tuple[int, int, float, float, float, float]] = []
    best_photos: dict[int, tuple[int, float, float, object]] = {}
    scoring_candidates: list[dict] = []
    recent_roi_candidates: list[tuple[float, float, float]] = []
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
            if not identities_seeded and detections and t_s <= seed_assignment_deadline_s:
                max_distance = max(60.0, min(frame.shape[:2]) * 0.22)
                matches = _match_initial_seeds(seed_points, detections, set(seed_to_track),
                                               set(seed_to_track.values()), max_distance)
                for robot_id, (track_id, distance) in matches.items():
                    seed_to_track[robot_id] = track_id
                    logger.info("Seeded robot %d to detector track %d at %.1f px from click", robot_id, track_id, distance)
                identities_seeded = len(seed_to_track) == len(robot_ids)
                if matches:
                    logger.info("Seeded %d of %d identities during initial %.1fs window", len(seed_to_track), len(robot_ids), seed_assignment_deadline_s)
            for tid, x, y, conf, bbox in detections:
                robot_id = next((rid for rid, assigned in seed_to_track.items() if assigned == tid), None)
                if robot_id is None:
                    continue
                x_ft, y_ft = project_point(matrix, x, y)
                if not (0 <= x_ft <= 12 and 0 <= y_ft <= 12):
                    logger.info("Track %s outside calibrated field at %.2fs; skipped", robot_id, t_s)
                    continue
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
                        if not _new_roi_candidate(ball_x, ball_y, t_s, recent_roi_candidates):
                            continue
                        if robot_positions:
                            robot_id = min(robot_positions, key=lambda pos: (pos[1] - ball_x) ** 2 + (pos[2] - ball_y) ** 2)[0]
                        else:
                            robot_id = robot_ids[0]
                        scoring_candidates.append({"robot_id": robot_id, "t_s": t_s, "points": 0,
                                                   "confidence": "high" if ball_conf >= 0.7 else "medium",
                                                   "status": "flagged", "source": "goal_roi", "event_type": "unclassified"})
                        recent_roi_candidates.append((ball_x, ball_y, t_s))
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
        if tracker_override:
            Path(tracker_override.name).unlink(missing_ok=True)
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
    if len(seed_to_track) < len(robot_ids):
        logger.warning("Only %d of %d robot identities were safely matched within %.1fs. Unmatched seeds are omitted instead of guessed.",
                       len(seed_to_track), len(robot_ids), seed_assignment_deadline_s)
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
