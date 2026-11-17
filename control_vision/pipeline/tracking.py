from __future__ import annotations

import logging
import math
import time
import tempfile
from pathlib import Path
from typing import Callable

import cv2

from control_vision.image_files import write_image
from control_vision.pipeline.homography import matrix_from_corners, project_point
from control_vision.pipeline.identity_recovery import IdentityRecovery
from control_vision.pipeline.manual_tracking import _alliance_mask
from control_vision.pipeline.zones import point_in_polygon


def _new_roi_candidate(x: float, y: float, t_s: float,
                       recent: list[tuple[float, float, float]]) -> bool:
    """Avoid counting a briefly lost and reacquired goal element twice."""
    recent[:] = [item for item in recent if t_s - item[2] <= 2.0]
    return all((x - old_x) ** 2 + (y - old_y) ** 2 > 35 ** 2
               for old_x, old_y, _ in recent)


def _unmatched_roi_detections(detections, known_positions, known_track_ids=None,
                              max_distance: float = 35.0):
    """Return ROI detections that cannot be spatially matched to known elements.

    Detector output ordering is not stable across frames. Greedy nearest-neighbor
    matching is sufficient here because this ROI contains a small number of game
    elements and prevents an existing element at a different list index from being
    reported as newly arrived.
    """
    known_track_ids = known_track_ids or set()
    matched_detections = {
        index for index, detection in enumerate(detections)
        if len(detection) > 3 and detection[3] >= 0 and detection[3] in known_track_ids
    }
    pairs = []
    for detection_index, detection in enumerate(detections):
        if detection_index in matched_detections:
            continue
        x, y = detection[:2]
        for known_index, (known_x, known_y) in enumerate(known_positions):
            distance_sq = (x - known_x) ** 2 + (y - known_y) ** 2
            if distance_sq <= max_distance ** 2:
                pairs.append((distance_sq, detection_index, known_index))
    pairs.sort()
    matched_known = set()
    for _, detection_index, known_index in pairs:
        if detection_index not in matched_detections and known_index not in matched_known:
            matched_detections.add(detection_index)
            matched_known.add(known_index)
    return [detection for index, detection in enumerate(detections)
            if index not in matched_detections]


def _nearest_robot_id(positions, x: float, y: float, max_distance: float):
    if not positions:
        return None
    robot_id, robot_x, robot_y = min(
        positions, key=lambda pos: (pos[1] - x) ** 2 + (pos[2] - y) ** 2)
    return robot_id if (robot_x - x) ** 2 + (robot_y - y) ** 2 <= max_distance ** 2 else None


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


# A seed whose track vanishes inside the seed window moves to the one other
# box covering at least this much of the smaller of the two boxes.
SEED_RESEED_MIN_OVERLAP = 0.5


def _overlap_of_smaller(a, b) -> float:
    """Intersection area over the smaller box's area (1.0 when one box contains the other)."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller if smaller > 0 else 0.0


# A re-seed is tried on each sampled frame from its time until this much later.
RESEED_WINDOW_S = 1.0
# A click this close outside a box (in pixels) still selects it, so clicking a
# robot's floor contact point just under its box works.
RESEED_BOX_MARGIN_PX = 20.0


def _validate_reseeds(reseeds, robot_ids) -> list[tuple[int, float, float, float]]:
    """Check user re-seeds and return them sorted by time."""
    checked = []
    for item in reseeds or []:
        if len(item) != 4:
            raise ValueError(f"A re-seed needs robot id, time, x and y, not {item!r}.")
        robot_id, t_s, x, y = item
        if robot_id not in robot_ids:
            raise ValueError(f"Re-seed for robot {robot_id}, which has no seed.")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (t_s, x, y)) or t_s < 0:
            raise ValueError(f"Re-seed for robot {robot_id} needs a time of 0 s or later and a pixel position.")
        checked.append((int(robot_id), float(t_s), float(x), float(y)))
    return sorted(checked, key=lambda item: item[1])


def _apply_reseed(robot_id, x, y, t_s, detections, visible, seed_to_track, recovery, unowned_rows, rows,
                  frame_index, logger) -> bool:
    """Give ``robot_id`` the tracked box under a user's click. False if no tracked box is there.

    The click is the user's identity call, so it outranks the automatic rules: the
    robot is tracked again even after its recovery window closed, and a robot that
    held the clicked track loses it and becomes lost. Positions the track recorded
    while unowned, after the robot was last seen, are kept for the robot.
    """
    def hits(margin):
        return [(math.hypot(fx - x, fy - y), tid) for tid, fx, fy, _, (x1, y1, x2, y2) in detections
                if tid >= 0 and visible.get(tid) is not None
                and x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin]

    # A box the click is inside wins over one it is only near.
    candidates = hits(0.0) or hits(RESEED_BOX_MARGIN_PX)
    if not candidates:
        return False
    track_id = min(candidates)[1]
    if seed_to_track.get(robot_id) == track_id:
        # The user vouches for this identity now, so an expired robot can be re-linked again later.
        recovery.reseed(robot_id, track_id)
        logger.info("Re-seed of robot %d at %.2fs confirms its current track %d", robot_id, t_s, track_id)
        return True
    previous_owner = next((rid for rid, tid in seed_to_track.items() if tid == track_id), None)
    if previous_owner is not None:
        del seed_to_track[previous_owner]
        logger.warning("Robot %d released from track %d at %.2fs: the user re-seeded robot %d to it; robot %d "
                       "stays lost until re-seeded", previous_owner, track_id, t_s, robot_id, previous_owner)
    old_track = seed_to_track.get(robot_id)
    seed_to_track[robot_id] = track_id
    recovery.reseed(robot_id, track_id)
    last = recovery.last_seen.get(robot_id)
    kept = [row for row in unowned_rows.pop(track_id, [])
            if row[0] != frame_index and (last is None or row[1] > last.t_s)]
    rows.extend((robot_id, *row) for row in kept)
    logger.info("Re-seeded robot %d at %.2fs by the user: track %s -> %d; %d earlier positions kept",
                robot_id, t_s, "none" if old_track is None else old_track, track_id, len(kept))
    return True


# Highest class id that is a scoring element: 1-2 in the three-class schema,
# 1-3 in the four-class one (train_yolo.SUPPORTED_CLASSES). Class 0 is robot.
MAX_ELEMENT_CLASS_ID = 3
TRACKER_CONFIG = Path(__file__).resolve().parents[1] / "config" / "botsort.yaml"


def _alliance_fractions(frame, bbox) -> tuple[float, float] | None:
    """Fractions of red and blue alliance-colour pixels inside a detector box."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    area = float((x2 - x1) * (y2 - y1))
    return (cv2.countNonZero(_alliance_mask(hsv, "red")) / area,
            cv2.countNonZero(_alliance_mask(hsv, "blue")) / area)


def _tracker_config(base: dict, confidence_threshold: float, gmc_method: str | None = None,
                    reid_model_path: Path | None = None) -> dict:
    """Return the BoT-SORT settings for one run.

    In Ultralytics 8.4 BYTETracker/BOTSORT, detections scoring at least
    ``track_high_thresh`` take part in the first association, and only those
    can start a track, when they also reach ``new_track_thresh``. Detections
    above ``track_low_thresh`` but below ``track_high_thresh`` can only extend
    an already tracked object in the second, IoU-only association. The detector
    call already drops anything below the user's confidence, so with fixed 0.25
    thresholds a robot detected at 0.22 under a 0.20 setting could never start
    a track. Both thresholds follow the user's setting; the low threshold stays
    below it so the two association stages keep their order.
    """
    if not 0.0 < confidence_threshold < 1.0:
        raise ValueError(f"Detection confidence must be between 0 and 1, not {confidence_threshold}.")
    config = dict(base)
    config["track_high_thresh"] = confidence_threshold
    config["new_track_thresh"] = confidence_threshold
    config["track_low_thresh"] = min(float(base.get("track_low_thresh", 0.1)), confidence_threshold / 2)
    if gmc_method is not None:
        config["gmc_method"] = gmc_method
    if reid_model_path is not None:
        config["with_reid"] = True
        config["model"] = str(reid_model_path.resolve())
    return config


def _no_seed_match_message(seed_points, closest_tracked: dict[int, float], closest_untracked: dict[int, float],
                           gate_px: float | None, deadline_s: float) -> str:
    """Explain why no seed matched, separating tracked from untracked detections.

    Seeds can only match detections the tracker has given an ID. The tracker's
    track-start threshold follows the detection confidence setting, so a
    detection without an ID was too weak or too brief to start a track;
    lowering the confidence can help it, moving the click cannot.
    """
    gate = "" if gate_px is None else f" (a match needs one within {gate_px:.0f} px)"
    parts = []
    for robot_id, seed_x, seed_y in seed_points:
        tracked = closest_tracked.get(robot_id)
        untracked = closest_untracked.get(robot_id)
        if tracked is not None:
            nearest = f"nearest tracked robot {tracked:.0f} px away{gate}"
        elif untracked is not None:
            nearest = (f"a robot was detected {untracked:.0f} px away, but the tracker never started a "
                       "track for it")
        else:
            nearest = "no robot detected"
        parts.append(f"seed at ({seed_x:.0f}, {seed_y:.0f}): {nearest}")
    hint = ""
    if any(robot_id not in closest_tracked and robot_id in closest_untracked for robot_id, _, _ in seed_points):
        hint = (" For a robot detected without a track, you can also lower the detection confidence in Settings: "
                "the tracker then starts tracks from weaker detections, at the cost of more false detections.")
    return (f"No seeded robot was matched to a tracked robot in the first {deadline_s:.0f} seconds, so no "
            f"positions could be recorded and processing stopped early ({'; '.join(parts)}). Seed robots that "
            "are clearly visible at the start of the video and click near each robot's floor contact point."
            + hint)


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
    gmc_method: str | None = None,
    alliances: dict[int, str] | None = None,
    reseeds: list[tuple[int, float, float, float]] | None = None,
) -> tuple[list[tuple[int, int, float, float, float, float]], list[tuple[int, Path, float]], list[dict]]:
    """Run local YOLO tracking at the requested sampling rate and map robot footpoints.

    Each seed is (database robot id, seed x pixel, seed y pixel). Seeds are associated
    to the closest detector track in the opening window; afterward tracker IDs persist.
    When a seeded robot's track ends, ``IdentityRecovery`` may re-link it to one
    new track under strict one-to-one gating; otherwise the robot stays lost.
    ``gmc_method`` overrides the configured BoT-SORT global motion compensation.
    ``alliances`` maps robot id to "red" or "blue"; re-linking is off without it.
    ``reseeds`` are a user's mid-match corrections, each (robot id, time s, x pixel,
    y pixel): see ``_apply_reseed``.
    """
    pending_reseeds = _validate_reseeds(reseeds, [seed[0] for seed in seed_points])
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
    reid_path = Path(reid_model_path) if reid_model_path else None
    if reid_path is not None and not reid_path.is_file():
        cap.release()
        raise RuntimeError(f"Configured ReID model was not found at {reid_path}.")
    tracker_override = None
    try:
        import yaml
        config = _tracker_config(yaml.safe_load(TRACKER_CONFIG.read_text(encoding="utf-8")),
                                 confidence_threshold, gmc_method, reid_path)
        tracker_override = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False)
        yaml.safe_dump(config, tracker_override)
        tracker_override.close()
        tracker_cfg = Path(tracker_override.name)
    except Exception as exc:
        cap.release()
        if tracker_override:
            Path(tracker_override.name).unlink(missing_ok=True)
        raise RuntimeError(f"Could not configure the tracker: {exc}") from exc
    logger.info("BoT-SORT thresholds: high %.2f, new track %.2f, low %.2f; global motion compensation: %s",
                config["track_high_thresh"], config["new_track_thresh"], config["track_low_thresh"],
                config.get("gmc_method"))
    if reid_path is not None:
        logger.info("BoT-SORT appearance ReID enabled using local model: %s", reid_path)
    else:
        logger.warning("BoT-SORT appearance ReID is disabled; using motion and overlap association only.")
    try:
        model = YOLO(str(model_path), task="detect")
        matrix = matrix_from_corners(corners)
    except Exception:
        cap.release()
        if tracker_override:
            Path(tracker_override.name).unlink(missing_ok=True)
        raise
    seed_to_track: dict[int, int] = {}
    # Last box of each seeded robot inside the seed window.
    seed_boxes: dict[int, tuple[float, float, float, float]] = {}
    recovery_alliances = {rid: str(value).strip().lower() for rid, value in (alliances or {}).items()}
    recovery = IdentityRecovery(alliances=recovery_alliances)
    if not alliances:
        logger.warning("No robot alliances were given, so lost robots are not re-linked to new tracks.")
    recoveries = 0
    # Field rows of tracks no robot owns yet, so a re-linked track keeps the
    # positions it recorded between its birth and the re-link.
    unowned_rows: dict[int, list[tuple[int, float, float, float, float]]] = {}
    identities_seeded = False
    seed_assignment_deadline_s = 3.0
    # Closest detection to each seed click during the assignment window,
    # reported if no seed can be matched.
    closest_tracked: dict[int, float] = {}
    closest_untracked: dict[int, float] = {}
    seed_gate_px: float | None = None
    robot_ids = [seed[0] for seed in seed_points]
    rows: list[tuple[int, int, float, float, float, float]] = []
    best_photos: dict[int, tuple[int, float, float, object]] = {}
    scoring_candidates: list[dict] = []
    recent_roi_candidates: list[tuple[float, float, float]] = []
    sampled = 0
    frame_index = 0
    empty = 0
    consecutive_inference_failures = 0
    stable_roi_count = 0
    stable_roi_positions: list[tuple[float, float]] = []
    stable_roi_track_ids: set[int] = set()
    roi_baseline_set = False
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
            if not seed_to_track and t_s > seed_assignment_deadline_s:
                # Identities are only assigned in the opening window, so a run
                # with no matched seed can never record a position. Stop now
                # instead of processing the rest of the match for nothing.
                raise RuntimeError(_no_seed_match_message(seed_points, closest_tracked, closest_untracked,
                                                          seed_gate_px, seed_assignment_deadline_s))
            try:
                result = model.track(frame, persist=True, tracker=str(tracker_cfg), imgsz=640, conf=confidence_threshold, iou=0.5, verbose=False)[0]
            except Exception as exc:
                logger.exception("Detection failed on frame %d; skipped", frame_index)
                consecutive_inference_failures += 1
                if consecutive_inference_failures >= 3:
                    raise RuntimeError("Robot model failed on three consecutive sampled frames. Check the model and runtime.") from exc
                frame_index += 1
                sampled += 1
                progress("Detect and track", min(100.0, 100 * sampled / sampled_total), f"Skipped unreadable frame at {t_s:.1f}s")
                continue
            consecutive_inference_failures = 0
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
                    elif 1 <= class_id <= MAX_ELEMENT_CLASS_ID and goal_roi and point_in_polygon(center_x, center_y, goal_roi):
                        # Classes 1..3 are scoring elements (POLLEN and NECTAR,
                        # or yellow/red/blue in the four-class schema); any
                        # other class an unsupported model emits is ignored.
                        track_id = int(box.id[0]) if box.id is not None else -1
                        roi_balls.append((center_x, center_y, conf, track_id))
            if sampled < 30 and not detections:
                empty += 1
            if sampled == 29 and empty == 30:
                raise RuntimeError("No robots were detected in the first 30 sampled frames. Check the video and robot model.")
            logger.info("Sampled frame %d (%.3fs): %d robot detections; %d scoring elements in ROI", frame_index, t_s, len(detections), len(roi_balls))
            if detections and t_s <= seed_assignment_deadline_s:
                # A seeded track that vanishes inside the window was usually a
                # second, briefly detected box on the same robot. Re-seed that
                # robot to a box covering the same place; otherwise its real
                # track, alive since before the robot was "last seen", could
                # never be re-linked later.
                visible_ids = {det[0] for det in detections if det[0] >= 0}
                for robot_id, old_track in sorted(seed_to_track.items()):
                    if old_track in visible_ids or robot_id not in seed_boxes:
                        continue
                    overlapping = [det[0] for det in detections
                                   if det[0] >= 0 and det[0] not in recovery.ever_assigned
                                   and _overlap_of_smaller(det[4], seed_boxes[robot_id]) >= SEED_RESEED_MIN_OVERLAP]
                    # Exactly one candidate, or the robot keeps its seed.
                    if len(overlapping) == 1:
                        seed_to_track[robot_id] = overlapping[0]
                        recovery.note_assignment(overlapping[0])
                        recovery.forget_robot(robot_id)
                        logger.info("Re-seeded robot %d from track %d, which vanished within the seed window, "
                                    "to overlapping track %d", robot_id, old_track, overlapping[0])
                if not identities_seeded:
                    max_distance = max(60.0, min(frame.shape[:2]) * 0.22)
                    seed_gate_px = max_distance
                    for robot_id, seed_x, seed_y in seed_points:
                        for track_id, x, y, *_ in detections:
                            distance = ((x - seed_x) ** 2 + (y - seed_y) ** 2) ** 0.5
                            # The matcher only considers detections with a tracker ID.
                            closest = closest_tracked if track_id >= 0 else closest_untracked
                            closest[robot_id] = min(distance, closest.get(robot_id, distance))
                    matches = _match_initial_seeds(seed_points, detections, set(seed_to_track),
                                                   recovery.ever_assigned, max_distance)
                    for robot_id, (track_id, distance) in matches.items():
                        seed_to_track[robot_id] = track_id
                        recovery.note_assignment(track_id)
                        logger.info("Seeded robot %d to detector track %d at %.1f px from click", robot_id, track_id, distance)
                    identities_seeded = len(seed_to_track) == len(robot_ids)
                    if matches:
                        logger.info("Seeded %d of %d identities during initial %.1fs window", len(seed_to_track), len(robot_ids), seed_assignment_deadline_s)
                for robot_id, track_id in seed_to_track.items():
                    box = next((det[4] for det in detections if det[0] == track_id), None)
                    if box is not None:
                        seed_boxes[robot_id] = box
            visible: dict[int, tuple[float, float] | None] = {}
            boxes_by_track = {det[0]: det[4] for det in detections if det[0] >= 0}
            owned_now = set(seed_to_track.values())
            for tid, x, y, conf, _ in detections:
                if tid < 0:
                    continue
                x_ft, y_ft = project_point(matrix, x, y)
                in_field = 0 <= x_ft <= 12 and 0 <= y_ft <= 12
                visible[tid] = (x_ft, y_ft) if in_field else None
                # Automatic recovery only uses never-assigned tracks; a pending
                # re-seed can also take a track a robot was moved off.
                if in_field and tid not in owned_now and (tid not in recovery.ever_assigned or pending_reseeds):
                    unowned_rows.setdefault(tid, []).append((frame_index, t_s, x_ft, y_ft, conf))
            recovery.observe_tracks(t_s, visible)
            # Each due re-seed is tried on its own, until it applies or its window passes.
            for item in [item for item in pending_reseeds if item[1] <= t_s]:
                robot_id, click_t, click_x, click_y = item
                if t_s - click_t > RESEED_WINDOW_S:
                    logger.warning("Re-seed of robot %d at %.2fs was not applied: no tracked robot box at the "
                                   "click within %.1fs", robot_id, click_t, RESEED_WINDOW_S)
                    pending_reseeds.remove(item)
                elif _apply_reseed(robot_id, click_x, click_y, t_s, detections, visible, seed_to_track,
                                   recovery, unowned_rows, rows, frame_index, logger):
                    pending_reseeds.remove(item)
            if seed_to_track:
                owned = set(seed_to_track.values())
                colours = {}
                if any(tid not in visible for tid in owned):
                    # Only needed while a seeded robot is missing.
                    for tid in visible:
                        if tid not in owned:
                            fractions = _alliance_fractions(frame, boxes_by_track[tid])
                            if fractions is not None:
                                colours[tid] = fractions
                for item in recovery.recover(t_s, visible, seed_to_track, logger, colours):
                    seed_to_track[item.robot_id] = item.new_track_id
                    recovery.apply(item)
                    recoveries += 1
                    # The current frame is recorded below like any owned track.
                    backfill = [row for row in unowned_rows.pop(item.new_track_id, []) if row[0] != frame_index]
                    rows.extend((item.robot_id, *row) for row in backfill)
                    logger.info("Recovered robot %d at %.2fs: track %d -> new track %d after %.2fs missing; first "
                                "footpoint %.2f ft from last position (gate %.2f ft); %d earlier positions kept",
                                item.robot_id, t_s, item.old_track_id, item.new_track_id, item.absence_s,
                                item.distance_ft, item.gate_ft, len(backfill))
            owned_now = set(seed_to_track.values())
            for tid in [tid for tid, buffered in unowned_rows.items()
                        if tid in owned_now or (tid in recovery.ever_assigned and not pending_reseeds)
                        or t_s - buffered[-1][1] > recovery.rules.max_absence_s]:
                # Keep only buffers that could still be re-linked.
                del unowned_rows[tid]
            for tid, x, y, conf, bbox in detections:
                robot_id = next((rid for rid, assigned in seed_to_track.items() if assigned == tid), None)
                if robot_id is None:
                    continue
                position = visible.get(tid)
                if position is None:
                    logger.info("Track %s outside calibrated field at %.2fs; skipped", robot_id, t_s)
                    continue
                x_ft, y_ft = position
                colour = _alliance_fractions(frame, bbox) if recovery.wants_colour(robot_id) else None
                recovery.observe_robot(robot_id, t_s, x_ft, y_ft, colour)
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
            current_roi_positions = [(ball[0], ball[1]) for ball in roi_balls]
            current_roi_track_ids = {ball[3] for ball in roi_balls if ball[3] >= 0}
            if goal_roi and not roi_baseline_set:
                # Elements already in the ROI on the first analysed frame were
                # there before the video started; they are not arrivals.
                stable_roi_count = current_roi_count
                stable_roi_positions = current_roi_positions
                stable_roi_track_ids = current_roi_track_ids
                roi_baseline_set = True
            new_roi_balls = _unmatched_roi_detections(
                roi_balls, stable_roi_positions, stable_roi_track_ids)
            if (current_roi_count == stable_roi_count and not current_roi_track_ids
                    and not stable_roi_track_ids):
                # Without tracker IDs, a same-count replacement is
                # indistinguishable from motion. Avoid inventing a goal event.
                new_roi_balls = []
            if goal_roi and new_roi_balls:
                if len(new_roi_balls) != pending_roi_count:
                    pending_roi_count, pending_roi_since = len(new_roi_balls), t_s
                elif pending_roi_since is not None and t_s - pending_roi_since >= 0.5:
                    for ball_x, ball_y, ball_conf, _ in new_roi_balls:
                        if not _new_roi_candidate(ball_x, ball_y, t_s, recent_roi_candidates):
                            continue
                        robot_id = _nearest_robot_id(robot_positions, ball_x, ball_y,
                                                     max(100.0, min(frame.shape[:2]) * 0.25))
                        scoring_candidates.append({"robot_id": robot_id, "t_s": t_s, "points": 0,
                                                   "confidence": "high" if ball_conf >= 0.7 else "medium",
                                                   "status": "flagged", "source": "goal_roi", "event_type": "unclassified"})
                        recent_roi_candidates.append((ball_x, ball_y, t_s))
                    stable_roi_count = current_roi_count
                    stable_roi_positions = current_roi_positions
                    stable_roi_track_ids = current_roi_track_ids
                    pending_roi_count, pending_roi_since = 0, None
            elif goal_roi:
                if current_roi_count < stable_roi_count:
                    # Elements that leave the scoring ROI free up positions for
                    # the next arrival; they do not erase candidate history.
                    stable_roi_count = current_roi_count
                    stable_roi_positions = current_roi_positions
                    stable_roi_track_ids = current_roi_track_ids
                elif not new_roi_balls:
                    # Keep spatial fallback tracks current for models without
                    # persistent element IDs, so ordinary motion is not a birth.
                    stable_roi_positions = current_roi_positions
                    stable_roi_track_ids = current_roi_track_ids
                pending_roi_count, pending_roi_since = 0, None
            sampled += 1
            progress("Detect and track", min(100.0, 100 * sampled / sampled_total), f"Processed {sampled:,} sampled frames")
            frame_index += 1
    finally:
        cap.release()
        if tracker_override:
            Path(tracker_override.name).unlink(missing_ok=True)
    for robot_id, click_t, _, _ in pending_reseeds:
        logger.warning("Re-seed of robot %d at %.2fs was not applied: the video ended before a tracked robot box "
                       "was found at the click", robot_id, click_t)
    if sampled and sampled < 30 and empty == sampled:
        raise RuntimeError("No robots were detected in the first sampled frames. Check the video and robot model.")
    if not rows:
        raise RuntimeError("No usable robot tracks were produced. Check the video, calibration, model, and identity seeds.")
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
    logger.info("Sampled frames: %d; mapped track points: %d; identity recoveries: %d", sampled, len(rows), recoveries)
    for robot_id in robot_ids:
        reference = recovery.colour_reference(robot_id)
        if reference is not None:
            logger.info("Robot %d own alliance-colour reference: %.3f; other alliance's colour: %.3f", robot_id,
                        reference, recovery.other_colour_reference(robot_id) or 0.0)
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
            if write_image(path, crop):
                photo_rows.append((robot_id, path, t_s))
            else:
                logger.warning("Robot %s photo could not be written to %s", robot_id, path)
    return rows, photo_rows, scoring_candidates
