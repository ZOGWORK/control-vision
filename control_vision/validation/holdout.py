"""Measure Control Vision results against human-labeled holdout matches.

The evaluator reads what the app saves in its project database (videos,
calibrations, robots, tracks, cycles, scoring_events and clips) and compares it
with a human ground-truth file per match. See docs/HOLDOUT_VALIDATION.md for
the ground-truth format and the exact metric definitions.

No metric is reported as a pass unless it was measured on at least
MIN_HOLDOUT_MATCHES distinct holdout matches. A metric without data is
reported as "not_measured", never as a pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from control_vision import __version__
from control_vision.game.scoreboard import calculate_scoreboard
from control_vision.pipeline.homography import FIELD_FEET, matrix_from_corners, project_point
from control_vision.training.split_manifest import parse_source_match, validate_split_manifest


TRUTH_FORMAT_VERSION = 1
MIN_HOLDOUT_MATCHES = 5
MAX_ASSOCIATION_DISTANCE_FT = 1.5
EVENT_TIME_TOLERANCE_S = 1.0
CLIP_BEFORE_S = 3.0
CLIP_AFTER_S = 3.0
CLIP_BOUND_TOLERANCE_S = 0.15
CLIP_DURATION_TOLERANCE_S = 0.5
MIN_LANDMARK_CORNER_DISTANCE_PX = 5.0
OFFICIAL_SCORE_HOSTS = ("ftc-events.firstinspires.org",)
PREVIEW_TRACKING_MODES = frozenset({"seeded_optical_flow"})

TARGETS = {
    "mean_id_switches_per_match_max": 2.0,
    "track_completeness_min": 0.95,
    "per_robot_identity_aware_completeness_min": 0.95,
    "max_reprojection_error_px_below": 5.0,
    "cycle_within_one_match_fraction_min": 0.90,
    "clip_coverage_required": 1.0,
    "score_within_ten_percent_match_fraction_min": 0.80,
    "score_tolerance_fraction": 0.10,
    "robot_attribution_fraction_min": 0.80,
    "minimum_matches": MIN_HOLDOUT_MATCHES,
}

PASS, FAIL, NOT_MEASURED, INSUFFICIENT = "pass", "fail", "not_measured", "insufficient_matches"


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def assign(cost: list[list[float]], gate: float) -> list[tuple[int, int, float]]:
    """Hungarian assignment that maximizes gated pairs, then minimizes total cost.

    ``cost[i][j]`` is the distance between row i and column j; entries above
    ``gate`` (or infinite) cannot be paired. Returns (row, column, cost) pairs.
    """
    rows = len(cost)
    columns = len(cost[0]) if rows else 0
    if not rows or not columns:
        return []
    size = max(rows, columns)
    # Any assignment with one more gated pair costs less than any with fewer,
    # because every gated cost is at most ``gate``.
    blocked = gate * (size + 1) + 1.0
    matrix = [[blocked] * size for _ in range(size)]
    for i in range(rows):
        for j in range(columns):
            value = float(cost[i][j])
            if math.isfinite(value) and value <= gate:
                matrix[i][j] = value
    # O(n^3) Kuhn-Munkres with potentials (1-indexed).
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    owner = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        owner[0] = i
        column = 0
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            row = owner[column]
            delta = math.inf
            next_column = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                reduced = matrix[row - 1][j - 1] - u[row] - v[j]
                if reduced < minimum[j]:
                    minimum[j] = reduced
                    way[j] = column
                if minimum[j] < delta:
                    delta = minimum[j]
                    next_column = j
            for j in range(size + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            previous = way[column]
            owner[column] = owner[previous]
            column = previous
    pairs = []
    for j in range(1, size + 1):
        i = owner[j] - 1
        if i < rows and j - 1 < columns and matrix[i][j - 1] < blocked:
            pairs.append((i, j - 1, float(cost[i][j - 1])))
    return sorted(pairs)


def _distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x_ft"]) - float(b["x_ft"]), float(a["y_ft"]) - float(b["y_ft"]))


def match_tracks(truth_by_frame: dict[int, list[dict]], predicted_by_frame: dict[int, list[dict]],
                 gate: float = MAX_ASSOCIATION_DISTANCE_FT) -> dict[str, Any]:
    """CLEAR-MOT style association of truth robots to predicted tracks.

    Per frame, a truth robot keeps its previous predicted track while that
    track is still within ``gate`` feet; everything else is paired by Hungarian
    assignment within the gate. An ID switch is a change of the predicted track
    matched to a truth robot compared with its last matched track; a frame
    with no match (fragmentation) does not count as a switch by itself.
    """
    last_match: dict[int, int] = {}
    visible: dict[int, int] = defaultdict(int)
    matched: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    switches: dict[int, int] = defaultdict(int)
    unmatched_predictions = 0
    for frame in sorted(truth_by_frame):
        actual = truth_by_frame[frame]
        predicted = predicted_by_frame.get(frame, [])
        by_track = {int(row["team_number"]): index for index, row in enumerate(predicted)}
        pairs: list[tuple[int, int]] = []
        taken_truth: set[int] = set()
        taken_predicted: set[int] = set()
        for truth_index, robot in enumerate(actual):
            team = int(robot["team_number"])
            previous = last_match.get(team)
            if previous is None or previous not in by_track:
                continue
            predicted_index = by_track[previous]
            if _distance(robot, predicted[predicted_index]) <= gate:
                pairs.append((truth_index, predicted_index))
                taken_truth.add(truth_index)
                taken_predicted.add(predicted_index)
        free_truth = [index for index in range(len(actual)) if index not in taken_truth]
        free_predicted = [index for index in range(len(predicted)) if index not in taken_predicted]
        cost = [[_distance(actual[t], predicted[p]) for p in free_predicted] for t in free_truth]
        for row, column, _ in assign(cost, gate):
            pairs.append((free_truth[row], free_predicted[column]))
        unmatched_predictions += len(predicted) - len(pairs)
        for robot in actual:
            visible[int(robot["team_number"])] += 1
        for truth_index, predicted_index in pairs:
            team = int(actual[truth_index]["team_number"])
            track = int(predicted[predicted_index]["team_number"])
            matched[team] += 1
            correct[team] += int(team == track)
            if team in last_match and last_match[team] != track:
                switches[team] += 1
            last_match[team] = track
    return {"visible": dict(visible), "matched": dict(matched), "correct": dict(correct),
            "switches": dict(switches), "unmatched_predictions": unmatched_predictions}


def _check_official_source(url: Any) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("official_scores.source_url must name the official results page "
                         f"on {OFFICIAL_SCORE_HOSTS[0]}.")
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in OFFICIAL_SCORE_HOSTS or not parsed.path.strip("/"):
        raise ValueError("official_scores.source_url must be an https link to a match results page on "
                         f"{OFFICIAL_SCORE_HOSTS[0]}, got {url!r}.")
    return url.strip()


def _validate_truth(truth: dict) -> None:
    required = ("format_version", "match_label", "source_match", "project_db", "video_id", "sample_fps",
                "tracks", "cycles", "scoring_events", "official_scores", "calibration_landmarks")
    missing = [name for name in required if name not in truth]
    if missing:
        raise ValueError("Missing holdout truth fields: " + ", ".join(missing))
    if truth["format_version"] != TRUTH_FORMAT_VERSION:
        raise ValueError(f"Unsupported holdout truth format_version {truth['format_version']!r}; "
                         f"expected {TRUTH_FORMAT_VERSION}.")
    if not str(truth["match_label"]).strip():
        raise ValueError("Holdout truth needs a match label.")
    parse_source_match(str(truth["source_match"]))
    if not _finite(truth["sample_fps"]) or float(truth["sample_fps"]) <= 0:
        raise ValueError("sample_fps must be positive.")
    if not truth["tracks"]:
        raise ValueError("Holdout truth needs visible robot observations at sampled frames.")
    landmarks = truth["calibration_landmarks"]
    if len(landmarks) < 4:
        raise ValueError("Holdout truth needs four independent calibration landmarks.")
    if not all(all(_finite(row.get(key)) for key in ("x_ft", "y_ft", "pixel_x", "pixel_y"))
               for row in landmarks):
        raise ValueError("Calibration landmarks need finite x_ft, y_ft, pixel_x and pixel_y.")
    landmark_points = [(float(row["x_ft"]), float(row["y_ft"])) for row in landmarks]
    if len(landmark_points) != len(set(landmark_points)):
        raise ValueError("Calibration landmarks must have distinct field positions.")
    pixel_points = [(float(row["pixel_x"]), float(row["pixel_y"])) for row in landmarks]
    if len(pixel_points) != len(set(pixel_points)):
        raise ValueError("Calibration landmarks must have distinct pixel positions.")
    if cv2.contourArea(cv2.convexHull(np.asarray(landmark_points, dtype=np.float32))) < 1:
        raise ValueError("Calibration landmarks must span an area of the field.")
    if cv2.contourArea(cv2.convexHull(np.asarray(pixel_points, dtype=np.float32))) < 25:
        raise ValueError("Calibration landmarks must span an area of the image.")
    official = truth["official_scores"]
    if not isinstance(official, dict) or not all(color in official for color in ("red", "blue")):
        raise ValueError("Holdout truth needs official red and blue scores.")
    if any(official[color] is None for color in ("red", "blue")):
        raise ValueError("Both official alliance scores must be reviewed.")
    if any(type(official[color]) is not int or official[color] < 0 for color in ("red", "blue")):
        raise ValueError("Official scores must be nonnegative whole numbers.")
    _check_official_source(official.get("source_url"))
    for row in truth["tracks"]:
        if type(row.get("visible")) is not bool:
            raise ValueError("Every truth row needs a reviewed true/false visible flag.")
        if not row["visible"]:
            continue
        has_position = row.get("x_ft") is not None or row.get("y_ft") is not None
        has_box = row.get("box_px") is not None
        if has_position == has_box:
            raise ValueError("Every visible truth robot needs either field coordinates (x_ft, y_ft) "
                             "or a pixel box (box_px), not both.")
        if has_position:
            if not (_finite(row.get("x_ft")) and _finite(row.get("y_ft"))
                    and 0 <= float(row["x_ft"]) <= FIELD_FEET and 0 <= float(row["y_ft"]) <= FIELD_FEET):
                raise ValueError("Visible truth positions must be finite and inside the 12-foot field.")
        else:
            box = row["box_px"]
            if (not isinstance(box, list) or len(box) != 4 or not all(_finite(value) for value in box)
                    or not (box[0] < box[2] and box[1] < box[3])):
                raise ValueError("box_px must be [x1, y1, x2, y2] in source pixels with x1 < x2 and y1 < y2.")
    if any(row.get("count") is None for row in truth["cycles"]):
        raise ValueError("Every team needs a reviewed hand-counted cycle total.")
    if any(type(row["count"]) is not int or row["count"] < 0 for row in truth["cycles"]):
        raise ValueError("Hand-counted cycle totals must be nonnegative whole numbers.")
    if len(truth["cycles"]) != len({int(row["team_number"]) for row in truth["cycles"]}):
        raise ValueError("Truth cycle counts may list each team only once.")
    for event in truth["scoring_events"]:
        if not _finite(event.get("t_s")) or float(event["t_s"]) < 0:
            raise ValueError("Every truth scoring event needs a nonnegative t_s in seconds.")
        if event.get("team_number") is not None and type(event["team_number"]) is not int:
            raise ValueError("Truth scoring event team_number must be a whole number or null.")
        if "event_type" in event and not isinstance(event["event_type"], str):
            raise ValueError("Truth scoring event_type must be text when given.")
    keys = [(int(row["frame"]), int(row["team_number"])) for row in truth["tracks"]]
    if len(keys) != len(set(keys)):
        raise ValueError("A team may appear only once per truth frame.")


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _probe_video(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Holdout video is missing or unreadable: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError("Holdout video needs a readable frame rate and frame count.")
    return fps, frame_count


def _clip_duration(path: Path) -> float | None:
    if not path.is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not (math.isfinite(fps) and math.isfinite(frames) and fps > 0 and frames > 0):
        return None
    return frames / fps


def clip_window(t_s: float, duration_s: float) -> tuple[float, float]:
    """The brief's highlight window: event -3 s .. +3 s, clamped to the video."""
    return max(0.0, t_s - CLIP_BEFORE_S), min(duration_s, t_s + CLIP_AFTER_S)


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    # as_uri() percent-encodes '#', '?' and '%' so SQLite opens the named file.
    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def evaluate_case(case_file: Path) -> dict:
    """Score one labeled holdout match against the app's saved project database."""
    case_file = case_file.resolve()
    truth = json.loads(case_file.read_text(encoding="utf-8"))
    _validate_truth(truth)
    db_path = _resolve(case_file.parent, truth["project_db"])
    if not db_path.is_file():
        raise FileNotFoundError(f"Project database is missing: {db_path}")
    connection = _open_read_only(db_path)
    try:
        video_id = int(truth["video_id"])
        video = connection.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if video is None:
            raise ValueError(f"Video {video_id} does not exist in {db_path}.")
        if video["analysis_status"] != "ready":
            raise ValueError("The holdout video has an unfinished or failed analysis; process it successfully first.")
        robots = connection.execute("SELECT * FROM robots WHERE video_id=? ORDER BY id", (video_id,)).fetchall()
        predicted_tracks = [dict(row) for row in connection.execute(
            "SELECT t.frame,t.x_ft,t.y_ft,r.team_number FROM tracks t "
            "JOIN robots r ON r.id=t.robot_id WHERE r.video_id=? ORDER BY t.frame,r.id", (video_id,))]
        predicted_cycles = connection.execute(
            "SELECT r.team_number,COUNT(c.id) AS count FROM robots r LEFT JOIN cycles c ON c.robot_id=r.id "
            "WHERE r.video_id=? GROUP BY r.id", (video_id,)).fetchall()
        # Same selection as ProjectDatabase.events_for_video.
        events = connection.execute(
            "SELECT e.*,r.team_number,r.alliance,COALESCE(e.scoring_alliance,r.alliance) AS event_alliance "
            "FROM scoring_events e LEFT JOIN robots r ON r.id=e.robot_id "
            "WHERE e.video_id=? OR r.video_id=? ORDER BY e.t_s,e.id", (video_id, video_id)).fetchall()
        event_ids = [int(event["id"]) for event in events]
        clips = connection.execute(
            f"SELECT scoring_event_id,path,start_t_s,end_t_s FROM clips WHERE scoring_event_id IN "
            f"({','.join('?' * len(event_ids))}) ORDER BY id", event_ids).fetchall() if event_ids else []
        calibration = connection.execute(
            "SELECT corners_json FROM calibrations WHERE video_id=?", (video_id,)).fetchone()
    finally:
        connection.close()

    team_numbers = {int(row["team_number"]) for row in robots}
    if not robots or len(team_numbers) != len(robots):
        raise ValueError("Seeded team numbers must exist and be unique within a holdout video.")
    if {int(row["team_number"]) for row in truth["tracks"]} != team_numbers:
        raise ValueError("Truth track teams must exactly match seeded teams in the project.")
    if {int(row["team_number"]) for row in truth["cycles"]} != team_numbers:
        raise ValueError("Truth cycle counts must include every seeded team.")
    unknown_event_teams = {int(event["team_number"]) for event in truth["scoring_events"]
                           if event.get("team_number") is not None} - team_numbers
    if unknown_event_teams:
        raise ValueError(f"Truth scoring events name teams not seeded in the video: {sorted(unknown_event_teams)}")
    if calibration is None:
        raise ValueError("The project has no saved calibration for this video.")
    video_path = _resolve(db_path.parent, video["path"])
    source_fps, frame_count = _probe_video(video_path)
    duration_s = float(video["duration_s"]) if video["duration_s"] else frame_count / source_fps
    if any(float(event["t_s"]) > duration_s for event in truth["scoring_events"]):
        raise ValueError("A truth scoring event is after the end of the video.")
    official = {color: int(truth["official_scores"][color]) for color in ("red", "blue")}
    stored_official = {"red": video["official_red"], "blue": video["official_blue"]}
    for color in ("red", "blue"):
        if stored_official[color] is not None and int(stored_official[color]) != official[color]:
            raise ValueError(f"The project's stored official {color} score ({stored_official[color]}) "
                             f"disagrees with the sourced truth ({official[color]}).")

    corners = json.loads(calibration["corners_json"])
    matrix = matrix_from_corners(corners)

    # Tracking.
    stride = max(1, round(source_fps / float(truth["sample_fps"])))
    expected = {(frame, team) for frame in range(0, frame_count, stride) for team in team_numbers}
    labeled = {(int(row["frame"]), int(row["team_number"])) for row in truth["tracks"]}
    if labeled != expected:
        raise ValueError("Truth must label visibility for every seeded team at every sampled video frame.")
    truth_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in truth["tracks"]:
        if not row["visible"]:
            continue
        position = dict(row)
        if row.get("box_px") is not None:
            x1, _, x2, y2 = (float(value) for value in row["box_px"])
            # The app projects the bottom-center of each robot box.
            position["x_ft"], position["y_ft"] = project_point(matrix, (x1 + x2) / 2, y2)
        truth_by_frame[int(row["frame"])].append(position)
    if not truth_by_frame:
        raise ValueError("Holdout truth has no visible robot observations.")
    predicted_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in predicted_tracks:
        if int(row["frame"]) % stride:
            raise ValueError("The truth sample_fps does not match the saved track sampling stride.")
        predicted_by_frame[int(row["frame"])].append(row)
    association = match_tracks(truth_by_frame, predicted_by_frame)
    visible_count = sum(association["visible"].values())
    matched_count = sum(association["matched"].values())
    identity_correct = sum(association["correct"].values())

    # Calibration against held-out landmarks (never the fitted corners).
    if any(math.hypot(float(landmark["pixel_x"]) - float(corner[0]),
                      float(landmark["pixel_y"]) - float(corner[1])) < MIN_LANDMARK_CORNER_DISTANCE_PX
           for landmark in truth["calibration_landmarks"] for corner in corners):
        raise ValueError("Calibration landmarks must be independent of the fitted corners.")
    inverse = np.linalg.inv(matrix)
    reprojection_errors = []
    for landmark in truth["calibration_landmarks"]:
        field = np.array([[[float(landmark["x_ft"]), float(landmark["y_ft"])]]], dtype=np.float32)
        pixel = cv2.perspectiveTransform(field, inverse)[0, 0]
        reprojection_errors.append(float(math.hypot(pixel[0] - float(landmark["pixel_x"]),
                                                    pixel[1] - float(landmark["pixel_y"]))))

    # Cycles.
    predicted_cycle_counts = {int(row["team_number"]): int(row["count"]) for row in predicted_cycles}
    cycle_rows = {str(int(row["team_number"])): {
        "hand_count": int(row["count"]),
        "app_count": predicted_cycle_counts.get(int(row["team_number"]), 0),
        "difference": predicted_cycle_counts.get(int(row["team_number"]), 0) - int(row["count"]),
    } for row in sorted(truth["cycles"], key=lambda item: int(item["team_number"]))}

    # Highlights.
    clip_files: dict[str, str] = {}
    clips_by_event: dict[int, list[dict]] = defaultdict(list)
    for clip in clips:
        path = _resolve(db_path.parent, clip["path"])
        start, end = float(clip["start_t_s"]), float(clip["end_t_s"])
        decoded = _clip_duration(path)
        if decoded is not None:
            clip_files.setdefault(str(path.resolve()), _sha256(path))
        clips_by_event[int(clip["scoring_event_id"])].append(
            {"start_t_s": start, "end_t_s": end, "decoded_duration_s": decoded})

    def clip_ok(event: sqlite3.Row) -> tuple[bool, bool]:
        """(decodable clip present, clip matches the clamped -3 s..+3 s window)."""
        expected_start, expected_end = clip_window(float(event["t_s"]), duration_s)
        present = window = False
        for clip in clips_by_event[int(event["id"])]:
            decoded = clip["decoded_duration_s"]
            if decoded is None:
                continue
            present = True
            window = window or (
                abs(clip["start_t_s"] - expected_start) <= CLIP_BOUND_TOLERANCE_S
                and abs(clip["end_t_s"] - expected_end) <= CLIP_BOUND_TOLERANCE_S
                and abs(decoded - (expected_end - expected_start)) <= CLIP_DURATION_TOLERANCE_S)
        return present, window

    active_events = [event for event in events if event["status"] != "rejected"]
    active_clip_checks = [clip_ok(event) for event in active_events]
    truth_events = list(truth["scoring_events"])

    def event_cost(candidates: list[sqlite3.Row]) -> list[list[float]]:
        return [[abs(float(app["t_s"]) - float(actual["t_s"]))
                 if ("event_type" not in actual or app["event_type"] == actual["event_type"]) else math.inf
                 for app in candidates] for actual in truth_events]

    truth_to_active = {row: column for row, column, _ in
                       assign(event_cost(active_events), EVENT_TIME_TOLERANCE_S)}
    truth_with_clip = sum(active_clip_checks[column][1] for column in truth_to_active.values())

    # Scoring.
    disqualified = {alliance for alliance in ("red", "blue") if video[f"{alliance}_disqualified"]}
    scoreboard = calculate_scoreboard(events, robots, int(video["swarm_threshold"]),
                                      int(video["pollinator_1_threshold"]),
                                      int(video["pollinator_2_threshold"]), disqualified)
    video_points = {color: int(scoreboard["alliance_points"][color]) for color in ("red", "blue")}
    alliance_error: dict[str, float | None] = {}
    for color in ("red", "blue"):
        difference = abs(video_points[color] - official[color])
        alliance_error[color] = (difference / official[color] if official[color]
                                 else (0.0 if difference == 0 else None))
    within_tolerance = bool(scoreboard["match_resolved"]) and all(
        error is not None and error <= TARGETS["score_tolerance_fraction"] for error in alliance_error.values())

    scored_events = [event for event in events if event["status"] == "confirmed" and int(event["points"]) != 0]
    truth_to_scored = {row: column for row, column, _ in
                       assign(event_cost(scored_events), EVENT_TIME_TOLERANCE_S)}
    known_truth = [index for index, event in enumerate(truth_events) if event.get("team_number") is not None]
    attribution_correct = sum(
        1 for index in known_truth if index in truth_to_scored
        and scored_events[truth_to_scored[index]]["team_number"] is not None
        and int(scored_events[truth_to_scored[index]]["team_number"]) == int(truth_events[index]["team_number"]))
    unmatched_app_scored = len(scored_events) - len(truth_to_scored)
    attribution_denominator = len(known_truth) + unmatched_app_scored

    per_robot = {}
    for team in sorted(team_numbers):
        visible = association["visible"].get(team, 0)
        per_robot[str(team)] = {
            "visible_observations": visible,
            "matched_observations": association["matched"].get(team, 0),
            "correct_identity_observations": association["correct"].get(team, 0),
            "id_switches": association["switches"].get(team, 0),
            "completeness": _fraction(association["matched"].get(team, 0), visible),
            "identity_aware_completeness": _fraction(association["correct"].get(team, 0), visible),
        }
    inputs = {"case": _file_record(case_file), "project_db": _file_record(db_path),
              "video": _file_record(video_path),
              "clips": [{"path": path, "sha256": digest} for path, digest in sorted(clip_files.items())]}
    return {
        "match_label": str(truth["match_label"]).strip(),
        "source_match": ":".join(parse_source_match(str(truth["source_match"]))),
        "video_id": video_id, "tracking_mode": video["tracking_mode"], "inputs": inputs,
        "sample_fps": float(truth["sample_fps"]), "sampled_frame_count": len(expected) // len(team_numbers),
        "tracking": {"visible_observations": visible_count, "matched_observations": matched_count,
                     "correct_identity_observations": identity_correct,
                     "unmatched_predicted_observations": association["unmatched_predictions"],
                     "completeness": _fraction(matched_count, visible_count),
                     "identity_correct_fraction": _fraction(identity_correct, matched_count),
                     "identity_aware_completeness": _fraction(identity_correct, visible_count),
                     "id_switches": sum(association["switches"].values()),
                     "per_robot": per_robot},
        "calibration": {"landmarks": len(reprojection_errors),
                        "reprojection_errors_px": reprojection_errors,
                        "max_reprojection_error_px": max(reprojection_errors),
                        "rms_reprojection_error_px": float(np.sqrt(np.mean(np.square(reprojection_errors))))},
        "cycles": {"by_team": cycle_rows,
                   "within_one_for_all_teams": all(abs(row["difference"]) <= 1 for row in cycle_rows.values())},
        "highlights": {"active_events": len(active_events),
                       "active_events_with_clip": sum(present for present, _ in active_clip_checks),
                       "active_events_with_correct_window": sum(window for _, window in active_clip_checks),
                       "truth_events": len(truth_events),
                       "truth_events_matched_to_app_event": len(truth_to_active),
                       "truth_events_with_correct_clip": truth_with_clip},
        "scoring": {"official": official, "official_source_url": truth["official_scores"]["source_url"].strip(),
                    "video": video_points, "alliance_error_fraction": alliance_error,
                    "review_queue_clear": bool(scoreboard["match_resolved"]),
                    "within_ten_percent": within_tolerance,
                    "confirmed_scored_events": len(scored_events),
                    "truth_events_with_known_robot": len(known_truth),
                    "truth_events_with_unknown_robot": len(truth_events) - len(known_truth),
                    "matched_scored_events": len(truth_to_scored),
                    "unmatched_app_scored_events": unmatched_app_scored,
                    "correct_robot_attribution": attribution_correct,
                    "attribution_denominator": attribution_denominator,
                    "attribution_fraction": _fraction(attribution_correct, attribution_denominator)},
    }


def _status(measured_matches: int, ok: bool | None) -> str:
    if ok is None or measured_matches == 0:
        return NOT_MEASURED
    if not ok:
        return FAIL
    return PASS if measured_matches >= MIN_HOLDOUT_MATCHES else INSUFFICIENT


def _combine(*statuses: str) -> str:
    if FAIL in statuses:
        return FAIL
    if NOT_MEASURED in statuses:
        return NOT_MEASURED
    if INSUFFICIENT in statuses:
        return INSUFFICIENT
    return PASS


def check_training_disjoint(cases: list[dict], dataset_root: Path, manifest: Path | None = None) -> dict:
    """Audit the training split manifest and reject holdout matches from training events.

    Uses the event-level holdout rule of control_vision.training.split_manifest:
    no train or val frame may come from a holdout event, even another match.
    """
    summary = validate_split_manifest(dataset_root, manifest)
    root = dataset_root.resolve()
    manifest_path = (manifest if manifest is not None and manifest.is_absolute()
                     else root / (manifest or "split_manifest.csv")).resolve()
    training_events: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"].strip() in {"train", "val"}:
                training_events.add(parse_source_match(row["source_match"])[0])
    leaks = sorted({case["source_match"] for case in cases
                    if case["source_match"].partition(":")[0] in training_events})
    if leaks:
        raise ValueError("Holdout matches come from events used for training or validation: " + ", ".join(leaks))
    return {"verified": True, "manifest": str(manifest_path), "manifest_sha256": summary["manifest_sha256"],
            "training_events": sorted(training_events),
            "holdout_events_in_manifest": summary["holdout_events"]}


def load_holdout_list(list_file: Path) -> tuple[list[Path], dict]:
    """Read a frozen holdout list and verify every case file's SHA-256."""
    list_file = list_file.resolve()
    data = json.loads(list_file.read_text(encoding="utf-8"))
    entries = data.get("cases") if isinstance(data, dict) else None
    if not entries:
        raise ValueError("The holdout list needs a nonempty 'cases' array of {path, sha256}.")
    paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError("Every holdout list entry needs path and sha256.")
        path = _resolve(list_file.parent, entry["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Holdout case is missing: {path}")
        digest = _sha256(path)
        if digest != str(entry["sha256"]).lower():
            raise ValueError(f"Holdout case {path} changed after the list was frozen "
                             f"(sha256 {digest}, list says {entry['sha256']}).")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("The holdout list names a case file more than once.")
    return paths, {"verified": True, "path": str(list_file), "sha256": _sha256(list_file),
                   "cases": len(paths)}


def aggregate(cases: list[dict], training_disjointness: dict | None = None,
              holdout_list: dict | None = None) -> dict:
    """Combine per-match results into phase statuses against the brief's targets."""
    if not cases:
        raise ValueError("At least one holdout case is required.")
    for key, message in (("match_label", "Holdout match labels must be unique."),
                         ("source_match", "Each holdout match (EVENT:MATCH) may be counted only once.")):
        values = [case[key] for case in cases]
        if len(values) != len(set(values)):
            raise ValueError(message)
    for name, message in (("video", "The same source video may be counted only once."),
                          ("case", "The same truth file may be counted only once.")):
        digests = [case["inputs"][name]["sha256"] for case in cases]
        if len(digests) != len(set(digests)):
            raise ValueError(message)
    count = len(cases)

    visible = sum(case["tracking"]["visible_observations"] for case in cases)
    matched = sum(case["tracking"]["matched_observations"] for case in cases)
    identity_correct = sum(case["tracking"]["correct_identity_observations"] for case in cases)
    per_robot = [metrics["identity_aware_completeness"]
                 for case in cases for metrics in case["tracking"]["per_robot"].values()]
    # A seeded robot never labeled visible has no measurement; it cannot pass.
    per_robot_min = min((value if value is not None else 0.0) for value in per_robot)
    mean_switches = sum(case["tracking"]["id_switches"] for case in cases) / count
    completeness = _fraction(matched, visible)
    max_reprojection = max(case["calibration"]["max_reprojection_error_px"] for case in cases)
    phase_1 = _status(count, completeness is not None and (
        mean_switches <= TARGETS["mean_id_switches_per_match_max"]
        and completeness >= TARGETS["track_completeness_min"]
        and per_robot_min >= TARGETS["per_robot_identity_aware_completeness_min"]
        and max_reprojection < TARGETS["max_reprojection_error_px_below"]))

    cycle_fraction = _fraction(sum(case["cycles"]["within_one_for_all_teams"] for case in cases), count)
    phase_2 = _status(count, cycle_fraction is not None
                      and cycle_fraction >= TARGETS["cycle_within_one_match_fraction_min"])

    active = sum(case["highlights"]["active_events"] for case in cases)
    active_ok = sum(case["highlights"]["active_events_with_correct_window"] for case in cases)
    truth_events = sum(case["highlights"]["truth_events"] for case in cases)
    truth_ok = sum(case["highlights"]["truth_events_with_correct_clip"] for case in cases)
    clip_coverage = _fraction(active_ok, active)
    truth_clip_coverage = _fraction(truth_ok, truth_events)
    highlight_matches = sum(1 for case in cases
                            if case["highlights"]["active_events"] or case["highlights"]["truth_events"])
    phase_3 = _status(highlight_matches, None if not (active or truth_events) else (
        active_ok == active and truth_ok == truth_events))

    score_fraction = _fraction(sum(case["scoring"]["within_ten_percent"] for case in cases), count)
    attributed = sum(case["scoring"]["correct_robot_attribution"] for case in cases)
    attribution_denominator = sum(case["scoring"]["attribution_denominator"] for case in cases)
    attribution = _fraction(attributed, attribution_denominator)
    attribution_matches = sum(1 for case in cases if case["scoring"]["attribution_denominator"])
    phase_4 = _combine(
        _status(count, score_fraction is not None
                and score_fraction >= TARGETS["score_within_ten_percent_match_fraction_min"]),
        _status(attribution_matches, None if attribution is None
                else attribution >= TARGETS["robot_attribution_fraction_min"]))

    preview_used = any(case["tracking_mode"] in PREVIEW_TRACKING_MODES for case in cases)
    disjoint = training_disjointness or {
        "verified": False, "reason": "No training split manifest was given (--dataset-root)."}
    frozen = holdout_list or {"verified": False, "reason": "No frozen holdout list was given (--holdout-list)."}
    phases = {"phase_1_tracking": phase_1, "phase_2_cycles": phase_2,
              "phase_3_highlights": phase_3, "phase_4_scoring": phase_4}
    blockers = [f"{name}: {status}" for name, status in phases.items() if status != PASS]
    if count < MIN_HOLDOUT_MATCHES:
        blockers.append(f"only {count} holdout match(es); at least {MIN_HOLDOUT_MATCHES} are required")
    if preview_used:
        blockers.append("the experimental seeded optical-flow preview produced at least one result")
    if not disjoint.get("verified"):
        blockers.append("holdout/training disjointness not verified")
    if not frozen.get("verified"):
        blockers.append("holdout set not frozen by a hashed list")
    summary = {
        "match_count": count,
        "minimum_matches": MIN_HOLDOUT_MATCHES,
        "match_list": [case["match_label"] for case in cases],
        "source_matches": [case["source_match"] for case in cases],
        "preview_mode_used": preview_used,
        "targets": TARGETS,
        "tracking": {"mean_id_switches_per_match": mean_switches,
                     "id_switches_total": sum(case["tracking"]["id_switches"] for case in cases),
                     "visible_observations": visible, "matched_observations": matched,
                     "track_completeness": completeness,
                     "identity_aware_completeness": _fraction(identity_correct, visible),
                     "minimum_per_robot_identity_aware_completeness": per_robot_min,
                     "max_reprojection_error_px": max_reprojection, "status": phase_1},
        "cycles": {"matches_within_one": sum(case["cycles"]["within_one_for_all_teams"] for case in cases),
                   "matches": count, "cycle_within_one_match_fraction": cycle_fraction, "status": phase_2},
        "highlights": {"active_events": active, "active_events_with_correct_clip": active_ok,
                       "clip_coverage": clip_coverage, "truth_events": truth_events,
                       "truth_events_with_correct_clip": truth_ok,
                       "truth_event_clip_coverage": truth_clip_coverage,
                       "matches_with_events": highlight_matches, "status": phase_3},
        "scoring": {"matches_within_ten_percent": sum(case["scoring"]["within_ten_percent"] for case in cases),
                    "matches": count, "score_within_ten_percent_match_fraction": score_fraction,
                    "correct_robot_attribution": attributed,
                    "attribution_denominator": attribution_denominator,
                    "robot_attribution_fraction": attribution,
                    "matches_with_attribution_evidence": attribution_matches, "status": phase_4},
        "performance": {"status": NOT_MEASURED,
                        "reason": "The project database does not record processing time; measure the "
                                  "2:30 match at 8 fps (<= 10 min) budget on the reference laptop separately."},
        "phases": phases,
        "training_disjointness": disjoint,
        "holdout_list": frozen,
        "acceptance_targets_met": not blockers,
        "blockers": blockers,
        "per_match": [{"match_label": case["match_label"], "source_match": case["source_match"],
                       "id_switches": case["tracking"]["id_switches"],
                       "track_completeness": case["tracking"]["completeness"],
                       "max_reprojection_error_px": case["calibration"]["max_reprojection_error_px"],
                       "cycles_within_one": case["cycles"]["within_one_for_all_teams"],
                       "score_within_ten_percent": case["scoring"]["within_ten_percent"],
                       "attribution_fraction": case["scoring"]["attribution_fraction"]} for case in cases],
    }
    return {"harness": {"control_vision_version": __version__, "truth_format_version": TRUTH_FORMAT_VERSION,
                        "association_gate_ft": MAX_ASSOCIATION_DISTANCE_FT,
                        "event_time_tolerance_s": EVENT_TIME_TOLERANCE_S,
                        "clip_bound_tolerance_s": CLIP_BOUND_TOLERANCE_S,
                        "clip_duration_tolerance_s": CLIP_DURATION_TOLERANCE_S},
            "summary": summary, "cases": cases}


def make_template(db_path: Path, video_id: int, sample_fps: float) -> dict:
    """An unreviewed truth file listing every seeded team at every sampled frame."""
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive.")
    connection = _open_read_only(db_path)
    try:
        video = connection.execute("SELECT path,event_code,match_number FROM videos WHERE id=?",
                                   (video_id,)).fetchone()
        if video is None:
            raise ValueError(f"Video {video_id} does not exist in {db_path}.")
        teams = [int(row[0]) for row in connection.execute(
            "SELECT team_number FROM robots WHERE video_id=? ORDER BY id", (video_id,))]
    finally:
        connection.close()
    if not teams:
        raise ValueError("Seed robot identities before making a holdout template.")
    fps, frame_count = _probe_video(_resolve(db_path.parent, video["path"]))
    stride = max(1, round(fps / sample_fps))
    source_match = (f"{video['event_code']}:{video['match_number']}"
                    if video["event_code"] and video["match_number"] else "")
    return {
        "format_version": TRUTH_FORMAT_VERSION, "match_label": "", "source_match": source_match,
        "project_db": str(db_path.resolve()), "video_id": video_id, "sample_fps": sample_fps,
        "tracks": [{"frame": frame, "team_number": team, "visible": None, "x_ft": None, "y_ft": None}
                   for frame in range(0, frame_count, stride) for team in teams],
        "cycles": [{"team_number": team, "count": None} for team in teams],
        "scoring_events": [],
        "official_scores": {"red": None, "blue": None, "source_url": None},
        "calibration_landmarks": [],
    }


def write_output(path: Path, text: str, replace: bool = False) -> None:
    """Write ``text`` atomically; never replace an existing file unless ``replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        if replace:
            os.replace(temporary, path)
        else:
            # A hard link fails if the target exists, so a file created after
            # the caller's check is still never overwritten.
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise
            except OSError:
                # No hard links on this volume: fall back to exclusive create.
                # FileExistsError from open("x") leaves someone else's file alone;
                # a failed write removes only the file this call created.
                stream = path.open("x", encoding="utf-8")
                try:
                    with stream:
                        stream.write(text)
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Control Vision holdout matches")
    parser.add_argument("cases", type=Path, nargs="*", help="Human-labeled holdout JSON files")
    parser.add_argument("--holdout-list", type=Path, help="Frozen holdout list of case paths and SHA-256s")
    parser.add_argument("--dataset-root", type=Path, help="Training dataset root with split_manifest.csv")
    parser.add_argument("--manifest", type=Path, help="Split manifest path (relative to --dataset-root)")
    parser.add_argument("--output", type=Path,
                        help="Write the JSON report or template here (refuses an existing file)")
    parser.add_argument("--force", action="store_true", help="Allow --output to replace an existing file")
    parser.add_argument("--make-template", action="store_true", help="Create an unreviewed label template")
    parser.add_argument("--project", type=Path, help="Project database for a label template")
    parser.add_argument("--video-id", type=int, help="Video ID for a label template")
    parser.add_argument("--sample-fps", type=float, help="Sampling rate used by the app")
    args = parser.parse_args(argv)
    report: dict
    paths: list[Path] = list(args.cases)
    if args.make_template:
        if not (args.project and args.video_id is not None and args.sample_fps and args.output):
            parser.error("Template needs --project, --video-id, --sample-fps, and --output")
        if args.cases or args.holdout_list:
            parser.error("Do not pass case files when making a template")
        report = make_template(args.project, args.video_id, args.sample_fps)
    else:
        if args.manifest and not args.dataset_root:
            parser.error("--manifest needs --dataset-root")
        frozen = None
        if args.holdout_list:
            if paths:
                parser.error("Pass either case files or --holdout-list, not both")
            paths, frozen = load_holdout_list(args.holdout_list)
        if not paths:
            parser.error("Pass one or more human-labeled holdout cases or --holdout-list")
        cases = [evaluate_case(path) for path in paths]
        disjoint = (check_training_disjoint(cases, args.dataset_root, args.manifest)
                    if args.dataset_root else None)
        report = aggregate(cases, disjoint, frozen)
    data = json.dumps(report, indent=2)
    if args.output:
        output = args.output.resolve()
        # Inputs are never replaced, even with --force.
        protected = {path.resolve() for path in paths}
        for path in (args.holdout_list, args.project):
            if path is not None:
                protected.add(path.resolve())
        for case in report.get("cases", []) if not args.make_template else []:
            inputs = case["inputs"]
            protected.update(Path(inputs[name]["path"]) for name in ("case", "project_db", "video"))
            protected.update(Path(clip["path"]) for clip in inputs["clips"])
        manifest = (report.get("summary", {}).get("training_disjointness", {}).get("manifest")
                    if not args.make_template else None)
        if manifest:
            protected.add(Path(manifest))
        if output in {path.resolve() for path in protected}:
            parser.error(f"--output {output} is an input to this run; refusing to overwrite it")
        if output.exists() and not args.force:
            parser.error(f"--output {output} already exists; pass --force to replace it")
        write_output(output, replace=args.force, text=data + "\n")
    else:
        print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
