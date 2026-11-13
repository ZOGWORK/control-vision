"""Measure Control Vision results against manually labeled holdout matches.

Truth locations are field coordinates at the exact source frame numbers sampled
by the app. A label for every visible robot at every evaluated frame is required
for the coverage number to mean what it says.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from control_vision.game.scoreboard import calculate_scoreboard
from control_vision.pipeline.homography import matrix_from_corners


MAX_ASSOCIATION_DISTANCE_FT = 1.5
EVENT_TIME_TOLERANCE_S = 1.0


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _clip_usable(path: Path, expected_duration: float) -> bool:
    if not path.is_file():
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps > 0 and frames > 0 and frames / fps >= expected_duration - 0.25
    finally:
        capture.release()


def _best_pairs(truth: list[dict], predicted: list[dict], max_distance: float):
    """Maximize one-to-one matches, then minimize total field distance."""
    best: list[tuple[int, int, float]] = []

    def visit(index: int, used: set[int], pairs: list[tuple[int, int, float]]):
        nonlocal best
        if index == len(truth):
            if (len(pairs), -sum(pair[2] for pair in pairs)) > (
                len(best), -sum(pair[2] for pair in best)
            ):
                best = pairs.copy()
            return
        visit(index + 1, used, pairs)
        actual = truth[index]
        for candidate_index, candidate in enumerate(predicted):
            if candidate_index in used:
                continue
            distance = math.hypot(actual["x_ft"] - candidate["x_ft"],
                                  actual["y_ft"] - candidate["y_ft"])
            if distance <= max_distance:
                pairs.append((index, candidate_index, distance))
                visit(index + 1, used | {candidate_index}, pairs)
                pairs.pop()

    visit(0, set(), [])
    return best


def _validate_truth(truth: dict) -> None:
    required = ("match_label", "project_db", "video_id", "sample_fps", "tracks", "cycles",
                "scoring_events", "official_scores", "calibration_landmarks")
    missing = [name for name in required if name not in truth]
    if missing:
        raise ValueError("Missing holdout truth fields: " + ", ".join(missing))
    if not truth["tracks"]:
        raise ValueError("Holdout truth needs visible robot observations at sampled frames.")
    if not str(truth["match_label"]).strip():
        raise ValueError("Holdout truth needs a match label.")
    if len(truth["calibration_landmarks"]) < 4:
        raise ValueError("Holdout truth needs four independent calibration landmarks.")
    landmark_points = [(float(row["x_ft"]), float(row["y_ft"]))
                       for row in truth["calibration_landmarks"]]
    if len(landmark_points) != len(set(landmark_points)):
        raise ValueError("Calibration landmarks must have distinct field positions.")
    pixel_points = [(float(row["pixel_x"]), float(row["pixel_y"]))
                    for row in truth["calibration_landmarks"]]
    if len(pixel_points) != len(set(pixel_points)):
        raise ValueError("Calibration landmarks must have distinct pixel positions.")
    if cv2.contourArea(cv2.convexHull(np.asarray(landmark_points, dtype=np.float32))) < 1:
        raise ValueError("Calibration landmarks must span an area of the field.")
    if cv2.contourArea(cv2.convexHull(np.asarray(pixel_points, dtype=np.float32))) < 25:
        raise ValueError("Calibration landmarks must span an area of the image.")
    if not all(color in truth["official_scores"] for color in ("red", "blue")):
        raise ValueError("Holdout truth needs official red and blue scores.")
    if any(type(row.get("visible")) is not bool for row in truth["tracks"]):
        raise ValueError("Every truth row needs a reviewed true/false visible flag.")
    if any(row["visible"] and (row.get("x_ft") is None or row.get("y_ft") is None)
           for row in truth["tracks"]):
        raise ValueError("Every visible truth robot needs field coordinates.")
    if any(row["visible"] and not (math.isfinite(float(row["x_ft"]))
                                   and math.isfinite(float(row["y_ft"]))
                                   and 0 <= float(row["x_ft"]) <= 12
                                   and 0 <= float(row["y_ft"]) <= 12)
           for row in truth["tracks"]):
        raise ValueError("Visible truth positions must be finite and inside the 12-foot field.")
    if any(row.get("count") is None for row in truth["cycles"]):
        raise ValueError("Every team needs a reviewed hand-counted cycle total.")
    if any(type(row["count"]) is not int or row["count"] < 0 for row in truth["cycles"]):
        raise ValueError("Hand-counted cycle totals must be nonnegative whole numbers.")
    if len(truth["cycles"]) != len({int(row["team_number"]) for row in truth["cycles"]}):
        raise ValueError("Truth cycle counts may list each team only once.")
    if any(truth["official_scores"][color] is None for color in ("red", "blue")):
        raise ValueError("Both official alliance scores must be reviewed.")
    if any(type(truth["official_scores"][color]) is not int or truth["official_scores"][color] < 0
           for color in ("red", "blue")):
        raise ValueError("Official scores must be nonnegative whole numbers.")
    keys = [(int(row["frame"]), int(row["team_number"])) for row in truth["tracks"]]
    if len(keys) != len(set(keys)):
        raise ValueError("A team may appear only once per truth frame.")
    if float(truth["sample_fps"]) <= 0:
        raise ValueError("sample_fps must be positive.")


def evaluate_case(case_file: Path) -> dict:
    truth = json.loads(case_file.read_text(encoding="utf-8"))
    _validate_truth(truth)
    db_path = Path(truth["project_db"])
    if not db_path.is_absolute():
        db_path = case_file.parent / db_path
    if not db_path.is_file():
        raise FileNotFoundError(f"Project database is missing: {db_path}")
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        video_id = int(truth["video_id"])
        video = connection.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if video is None:
            raise ValueError(f"Video {video_id} does not exist in {db_path}.")
        if video["analysis_status"] != "ready":
            raise ValueError("The holdout video has an unfinished or failed analysis; process it successfully first.")
        robots = connection.execute("SELECT * FROM robots WHERE video_id=?", (video_id,)).fetchall()
        robot_by_id = {row["id"]: row for row in robots}
        team_numbers = {int(row["team_number"]) for row in robots}
        if len(team_numbers) != len(robots):
            raise ValueError("Seeded team numbers must be unique within a holdout video.")
        truth_teams = {int(row["team_number"]) for row in truth["tracks"]}
        if truth_teams != team_numbers:
            raise ValueError("Truth track teams must exactly match seeded teams in the project.")
        cycle_teams = {int(row["team_number"]) for row in truth["cycles"]}
        if cycle_teams != team_numbers:
            raise ValueError("Truth cycle counts must include every seeded team.")
        predicted_tracks = [dict(row) for row in connection.execute(
            "SELECT t.frame,t.x_ft,t.y_ft,r.team_number FROM tracks t "
            "JOIN robots r ON r.id=t.robot_id WHERE r.video_id=? ORDER BY t.frame", (video_id,))]
        predicted_cycles = connection.execute(
            "SELECT r.team_number,COUNT(*) AS count FROM cycles c JOIN robots r ON r.id=c.robot_id "
            "WHERE r.video_id=? GROUP BY r.team_number", (video_id,)).fetchall()
        events = connection.execute(
            "SELECT e.*,r.team_number FROM scoring_events e LEFT JOIN robots r ON r.id=e.robot_id "
            "WHERE e.video_id=? ORDER BY e.t_s", (video_id,)).fetchall()
        clips = connection.execute(
            "SELECT c.scoring_event_id,c.path,c.start_t_s,c.end_t_s FROM clips c "
            "JOIN scoring_events e ON e.id=c.scoring_event_id WHERE e.video_id=?", (video_id,)).fetchall()
        calibration = connection.execute(
            "SELECT corners_json FROM calibrations WHERE video_id=?", (video_id,)).fetchone()
    finally:
        connection.close()

    if calibration is None:
        raise ValueError("The project has no saved calibration for this video.")
    video_path = Path(video["path"])
    if not video_path.is_absolute():
        video_path = db_path.parent / video_path
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Holdout video is missing or unreadable: {video_path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if source_fps <= 0 or frame_count <= 0:
        raise ValueError("Holdout video needs a readable frame rate and frame count.")
    stride = max(1, round(source_fps / float(truth["sample_fps"])))
    expected = {(frame, team) for frame in range(0, frame_count, stride)
                for team in team_numbers}
    labeled = {(int(row["frame"]), int(row["team_number"])) for row in truth["tracks"]}
    if labeled != expected:
        raise ValueError("Truth must label visibility for every seeded team at every sampled video frame.")
    by_frame_truth: dict[int, list[dict]] = defaultdict(list)
    by_frame_predicted: dict[int, list[dict]] = defaultdict(list)
    for row in truth["tracks"]:
        if row.get("visible", True):
            by_frame_truth[int(row["frame"])].append(row)
    if not by_frame_truth:
        raise ValueError("Holdout truth has no visible robot observations.")
    for row in predicted_tracks:
        if int(row["frame"]) % stride:
            raise ValueError("The truth sample_fps does not match the saved track sampling stride.")
        by_frame_predicted[int(row["frame"])].append(row)
    visible_count = matched_count = identity_correct = id_switches = 0
    visible_by_team = defaultdict(int)
    matched_by_team = defaultdict(int)
    correct_by_team = defaultdict(int)
    previous_identity: dict[int, int] = {}
    for frame in sorted(by_frame_truth):
        actual, predicted = by_frame_truth[frame], by_frame_predicted[frame]
        visible_count += len(actual)
        for robot in actual:
            visible_by_team[int(robot["team_number"])] += 1
        for actual_index, predicted_index, _ in _best_pairs(
            actual, predicted, MAX_ASSOCIATION_DISTANCE_FT
        ):
            matched_count += 1
            team = int(actual[actual_index]["team_number"])
            assigned = int(predicted[predicted_index]["team_number"])
            matched_by_team[team] += 1
            correct_by_team[team] += int(team == assigned)
            identity_correct += int(team == assigned)
            if team in previous_identity and previous_identity[team] != assigned:
                id_switches += 1
            previous_identity[team] = assigned

    corners = json.loads(calibration["corners_json"])
    if any(math.hypot(float(landmark["pixel_x"]) - float(corner[0]),
                      float(landmark["pixel_y"]) - float(corner[1])) < 5
           for landmark in truth["calibration_landmarks"] for corner in corners):
        raise ValueError("Calibration landmarks must be independent of the fitted corners.")
    matrix = matrix_from_corners(corners)
    inverse = np.linalg.inv(matrix)
    reprojection_errors = []
    for landmark in truth["calibration_landmarks"]:
        field = np.array([[[float(landmark["x_ft"]), float(landmark["y_ft"])]]], dtype=np.float32)
        pixel = cv2.perspectiveTransform(field, inverse)[0, 0]
        reprojection_errors.append(math.hypot(pixel[0] - float(landmark["pixel_x"]),
                                               pixel[1] - float(landmark["pixel_y"])))

    predicted_cycle_counts = {int(row["team_number"]): int(row["count"])
                              for row in predicted_cycles}
    cycle_differences = {int(row["team_number"]): predicted_cycle_counts.get(
        int(row["team_number"]), 0) - int(row["count"])
        for row in truth["cycles"]}
    active_events = [event for event in events if event["status"] != "rejected"]
    clip_by_event: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for clip in clips:
        clip_by_event[int(clip["scoring_event_id"])].append(clip)
    clip_present = clip_window_correct = 0
    for event in active_events:
        expected_start = max(0.0, float(event["t_s"]) - 3.0)
        expected_end = min(float(video["duration_s"]), float(event["t_s"]) + 3.0)
        candidates = [clip for clip in clip_by_event[int(event["id"])]
                      if _clip_usable((db_path.parent / clip["path"]),
                                      expected_end - expected_start)]
        if candidates:
            clip_present += 1
        if any(float(clip["start_t_s"]) <= expected_start + 0.15
               and float(clip["end_t_s"]) >= expected_end - 0.15 for clip in candidates):
            clip_window_correct += 1

    disqualified = {alliance for alliance in ("red", "blue")
                    if video[f"{alliance}_disqualified"]}
    scoreboard = calculate_scoreboard(events, robots, int(video["swarm_threshold"]),
                                      int(video["pollinator_1_threshold"]),
                                      int(video["pollinator_2_threshold"]), disqualified)
    official = {color: int(truth["official_scores"][color]) for color in ("red", "blue")}
    video_points = scoreboard["alliance_points"]
    official_total = official["red"] + official["blue"]
    video_total = video_points["red"] + video_points["blue"]
    score_error_fraction = abs(video_total - official_total) / max(1, official_total)
    predicted_scored = [event for event in events if event["status"] == "confirmed"
                        and int(event["points"]) != 0]
    truth_scored = list(truth["scoring_events"])
    unmatched_truth = set(range(len(truth_scored)))
    attribution_correct = matched_events = 0
    for event in predicted_scored:
        candidates = [(abs(float(event["t_s"]) - float(actual["t_s"])), index)
                      for index, actual in enumerate(truth_scored) if index in unmatched_truth
                      and abs(float(event["t_s"]) - float(actual["t_s"])) <= EVENT_TIME_TOLERANCE_S
                      and ("event_type" not in actual or event["event_type"] == actual["event_type"])]
        if not candidates:
            continue
        _, index = min(candidates)
        unmatched_truth.remove(index)
        matched_events += 1
        expected_team = truth_scored[index].get("team_number")
        attribution_correct += int(expected_team is not None and event["team_number"] is not None
                                   and int(event["team_number"]) == int(expected_team))

    truth_clips_present = truth_clips_cover_window = 0
    used_events: set[int] = set()
    for actual in truth_scored:
        matches = [(abs(float(event["t_s"]) - float(actual["t_s"])), event)
                   for event in active_events if int(event["id"]) not in used_events
                   and abs(float(event["t_s"]) - float(actual["t_s"])) <= EVENT_TIME_TOLERANCE_S
                   and ("event_type" not in actual or event["event_type"] == actual["event_type"])]
        if not matches:
            continue
        event = min(matches, key=lambda pair: pair[0])[1]
        used_events.add(int(event["id"]))
        expected_start = max(0.0, float(actual["t_s"]) - 3.0)
        expected_end = min(float(video["duration_s"]), float(actual["t_s"]) + 3.0)
        candidates = [clip for clip in clip_by_event[int(event["id"])]
                      if _clip_usable(db_path.parent / clip["path"],
                                      expected_end - expected_start)]
        truth_clips_present += int(bool(candidates))
        truth_clips_cover_window += int(any(
            float(clip["start_t_s"]) <= expected_start + 0.15
            and float(clip["end_t_s"]) >= expected_end - 0.15 for clip in candidates))

    return {
        "match_label": truth["match_label"], "video_id": video_id,
        "video_path": str(video_path.resolve()), "project_db": str(db_path.resolve()),
        "tracking_mode": video["tracking_mode"],
        "sample_fps": float(truth["sample_fps"]), "sampled_frame_count": len(expected) // len(team_numbers),
        "tracking": {"visible_observations": visible_count, "matched_observations": matched_count,
                     "correct_identity_observations": identity_correct,
                     "completeness": _fraction(matched_count, visible_count),
                     "identity_correct_fraction": _fraction(identity_correct, matched_count),
                     "identity_aware_completeness": _fraction(identity_correct, visible_count),
                     "per_robot": {str(team): {
                         "visible_observations": visible_by_team[team],
                         "matched_observations": matched_by_team[team],
                         "correct_identity_observations": correct_by_team[team],
                         "completeness": _fraction(matched_by_team[team], visible_by_team[team]),
                         "identity_aware_completeness": _fraction(
                             correct_by_team[team], visible_by_team[team]),
                     } for team in sorted(team_numbers)},
                     "id_switches": id_switches,
                     "max_reprojection_error_px": max(reprojection_errors)},
        "cycles": {"differences_by_team": cycle_differences,
                   "within_one_for_all_teams": all(abs(value) <= 1 for value in cycle_differences.values())},
        "highlights": {"active_events": len(active_events), "clips_present": clip_present,
                       "clips_cover_three_seconds_each_side": clip_window_correct,
                       "truth_events": len(truth_scored),
                       "truth_events_with_clips": truth_clips_present,
                       "truth_clips_cover_three_seconds_each_side": truth_clips_cover_window},
        "scoring": {"official": official, "video": video_points,
                    "official_total": official_total, "video_total": video_total,
                    "total_error_fraction": score_error_fraction,
                    "review_queue_clear": scoreboard["match_resolved"],
                    "within_ten_percent": scoreboard["match_resolved"] and score_error_fraction <= 0.10,
                    "confirmed_scored_events": len(predicted_scored),
                    "matched_to_truth_events": matched_events,
                    "correct_robot_attribution": attribution_correct,
                    "attribution_fraction": _fraction(attribution_correct, len(predicted_scored))},
    }


def aggregate(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("At least one holdout case is required.")
    identities = [(case["project_db"], case["video_id"]) for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("Each holdout video may be counted only once.")
    paths = [case["video_path"] for case in cases]
    if len(paths) != len(set(paths)):
        raise ValueError("The same source video may be counted only once.")
    labels = [case["match_label"] for case in cases]
    if len(labels) != len(set(labels)):
        raise ValueError("Holdout match labels must be unique.")
    visible = sum(case["tracking"]["visible_observations"] for case in cases)
    matched = sum(case["tracking"]["matched_observations"] for case in cases)
    identity_correct = sum(case["tracking"]["correct_identity_observations"] for case in cases)
    per_robot_completeness = [metrics["identity_aware_completeness"] or 0.0
                              for case in cases for metrics in case["tracking"]["per_robot"].values()]
    scored = sum(case["scoring"]["confirmed_scored_events"] for case in cases)
    attributed = sum(case["scoring"]["correct_robot_attribution"] for case in cases)
    active = sum(case["highlights"]["active_events"] for case in cases)
    clips = sum(case["highlights"]["clips_cover_three_seconds_each_side"] for case in cases)
    truth_events = sum(case["highlights"]["truth_events"] for case in cases)
    truth_clips = sum(case["highlights"]["truth_clips_cover_three_seconds_each_side"]
                      for case in cases)
    summary = {
        "match_count": len(cases),
        "match_list": [case["match_label"] for case in cases],
        "preview_mode_used": any(case["tracking_mode"] == "seeded_optical_flow" for case in cases),
        "mean_id_switches_per_match": sum(case["tracking"]["id_switches"] for case in cases) / len(cases),
        "track_completeness": _fraction(matched, visible),
        "identity_aware_completeness": _fraction(identity_correct, visible),
        "minimum_per_robot_identity_aware_completeness": min(per_robot_completeness),
        "max_reprojection_error_px": max(case["tracking"]["max_reprojection_error_px"] for case in cases),
        "cycle_within_one_match_fraction": _fraction(sum(case["cycles"]["within_one_for_all_teams"]
                                                       for case in cases), len(cases)),
        "clip_coverage": _fraction(clips, active),
        "truth_event_clip_coverage": _fraction(truth_clips, truth_events),
        "score_within_ten_percent_match_fraction": _fraction(sum(case["scoring"]["within_ten_percent"]
                                                               for case in cases), len(cases)),
        "robot_attribution_fraction": _fraction(attributed, scored),
    }
    summary["phase_1_pass"] = (summary["mean_id_switches_per_match"] <= 2
                               and summary["track_completeness"] >= 0.95
                               and summary["minimum_per_robot_identity_aware_completeness"] >= 0.95
                               and summary["max_reprojection_error_px"] < 5)
    summary["phase_2_pass"] = summary["cycle_within_one_match_fraction"] >= 0.90
    summary["phase_3_pass"] = (active > 0 and truth_events > 0
                               and summary["clip_coverage"] == 1.0
                               and summary["truth_event_clip_coverage"] == 1.0)
    summary["phase_4_pass"] = (summary["score_within_ten_percent_match_fraction"] >= 0.80
                               and scored > 0 and summary["robot_attribution_fraction"] >= 0.80)
    summary["brief_targets_met"] = (len(cases) >= 5 and not summary["preview_mode_used"]
                                    and all(summary[f"phase_{phase}_pass"]
                                            for phase in range(1, 5)))
    return {"summary": summary, "cases": cases}


def make_template(db_path: Path, video_id: int, sample_fps: float) -> dict:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive.")
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        video = connection.execute("SELECT path FROM videos WHERE id=?", (video_id,)).fetchone()
        if video is None:
            raise ValueError(f"Video {video_id} does not exist in {db_path}.")
        teams = [int(row[0]) for row in connection.execute(
            "SELECT team_number FROM robots WHERE video_id=? ORDER BY id", (video_id,))]
    finally:
        connection.close()
    if not teams:
        raise ValueError("Seed robot identities before making a holdout template.")
    video_path = Path(video[0])
    if not video_path.is_absolute():
        video_path = db_path.parent / video_path
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Holdout video is missing or unreadable: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError("Holdout video needs a readable frame rate and frame count.")
    stride = max(1, round(fps / sample_fps))
    return {
        "match_label": "", "project_db": str(db_path.resolve()), "video_id": video_id,
        "sample_fps": sample_fps,
        "tracks": [{"frame": frame, "team_number": team, "visible": None,
                    "x_ft": None, "y_ft": None}
                   for frame in range(0, frame_count, stride) for team in teams],
        "cycles": [{"team_number": team, "count": None} for team in teams],
        "scoring_events": [], "official_scores": {"red": None, "blue": None},
        "calibration_landmarks": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Control Vision holdout matches")
    parser.add_argument("cases", type=Path, nargs="*", help="Human-labeled holdout JSON files")
    parser.add_argument("--output", type=Path, help="Write the JSON report here")
    parser.add_argument("--make-template", action="store_true", help="Create an unreviewed label template")
    parser.add_argument("--project", type=Path, help="Project database for a label template")
    parser.add_argument("--video-id", type=int, help="Video ID for a label template")
    parser.add_argument("--sample-fps", type=float, help="Sampling rate used by the app")
    args = parser.parse_args(argv)
    if args.make_template:
        if not (args.project and args.video_id is not None and args.sample_fps and args.output):
            parser.error("Template needs --project, --video-id, --sample-fps, and --output")
        if args.cases:
            parser.error("Do not pass case files when making a template")
        report = make_template(args.project, args.video_id, args.sample_fps)
    else:
        if not args.cases:
            parser.error("Pass one or more human-labeled holdout cases")
        report = aggregate([evaluate_case(path) for path in args.cases])
    data = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
