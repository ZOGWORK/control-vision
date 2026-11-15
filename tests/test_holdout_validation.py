import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from control_vision.storage.database import ProjectDatabase
from control_vision.validation.holdout import (
    MIN_HOLDOUT_MATCHES, aggregate, assign, check_training_disjoint, evaluate_case,
    load_holdout_list, main, make_template, match_tracks, write_output)

FPS = 4
SECONDS = 10
STRIDE = 2
RED, BLUE = 1234, 5678
OFFICIAL_URL = "https://ftc-events.firstinspires.org/2026/USMOSTLN/qualifications/1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_video(path: Path, frames: int, shade: int = 0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 160))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((160, 160, 3), shade, dtype=np.uint8))
    writer.release()


def make_case(folder: Path, *, index: int = 1, event_t: float = 5.0, clip_window=(2.0, 8.0),
              clip_seconds: float | None = None, tracks=None, truth_tracks=None,
              official=(2, 0), extra_truth_events=(), tracking_mode: str = "yolo") -> Path:
    """One synthetic holdout match: a real project database plus its truth file.

    Default: two robots tracked perfectly at every sampled frame, one confirmed
    2-point red event at 5 s with a correct 2..8 s clip, red 2 / blue 0 official.
    """
    folder.mkdir(parents=True, exist_ok=True)
    video_path = folder / "match.mp4"
    write_video(video_path, SECONDS * FPS, shade=index)
    db = ProjectDatabase(folder / "project.sqlite")
    project_id = db.add_project("holdout")
    video_id = db.add_video(project_id, video_path,
                            {"duration_s": float(SECONDS), "fps": FPS, "width": 160, "height": 160})
    db.save_calibration(video_id, [[0, 0], [120, 0], [120, 120], [0, 120]])
    red = db.add_robot(video_id, RED, "red", 20, 20)
    blue = db.add_robot(video_id, BLUE, "blue", 80, 80)
    frames = range(0, SECONDS * FPS, STRIDE)
    default = {frame: ((2.0, 2.0), (10.0, 10.0)) for frame in frames}
    rows = []
    for frame, positions in (default if tracks is None else tracks).items():
        for robot_id, position in zip((red, blue), positions):
            if position is not None:
                rows.append((robot_id, frame, frame / FPS, position[0], position[1], .9))
    db.replace_tracks(rows)
    event_id = db.add_scoring_event(red, event_t)
    db.review_event(event_id, "confirmed", 2, red, "flower_element", "high", "red")
    clip_path = folder / "clip.mp4"
    write_video(clip_path, round((clip_seconds or (clip_window[1] - clip_window[0])) * FPS))
    db.add_clip(event_id, clip_path, *clip_window)
    db.connection.execute("UPDATE videos SET tracking_mode=? WHERE id=?", (tracking_mode, video_id))
    db.connection.commit()
    db.set_analysis_status(video_id, "ready")
    db.close()
    labels = default if truth_tracks is None else truth_tracks
    truth = {
        "format_version": 1, "match_label": f"synthetic Q{index}", "source_match": f"USMOSTLN:Q{index}",
        "project_db": "project.sqlite", "video_id": video_id, "sample_fps": FPS / STRIDE,
        "tracks": [
            {"frame": frame, "team_number": team, "visible": position is not None,
             "x_ft": None if position is None else position[0],
             "y_ft": None if position is None else position[1]}
            for frame, positions in labels.items() for team, position in zip((RED, BLUE), positions)],
        "cycles": [{"team_number": RED, "count": 0}, {"team_number": BLUE, "count": 0}],
        "scoring_events": [{"t_s": event_t, "team_number": RED, "event_type": "flower_element"},
                           *extra_truth_events],
        "official_scores": {"red": official[0], "blue": official[1], "source_url": OFFICIAL_URL},
        "calibration_landmarks": [{"x_ft": x, "y_ft": y, "pixel_x": x * 10, "pixel_y": y * 10}
                                  for x, y in ((3, 3), (9, 3), (3, 9), (9, 9))],
    }
    case_path = folder / "case.json"
    case_path.write_text(json.dumps(truth), encoding="utf-8")
    return case_path


def edit_truth(case_path: Path, change) -> None:
    truth = json.loads(case_path.read_text(encoding="utf-8"))
    change(truth)
    case_path.write_text(json.dumps(truth), encoding="utf-8")


def make_dataset(root: Path, train_event: str = "USWASAM1") -> Path:
    rows = ["image,split,source_match,source,rights"]
    for split, match in (("train", f"{train_event}:Q1"), ("val", f"{train_event}:Q2")):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"{split}.jpg").write_bytes(split.encode())
        (root / "labels" / split / f"{split}.txt").write_text("", encoding="utf-8")
        rows.append(f"images/{split}/{split}.jpg,{split},{match},https://example.test/footage,team permission")
    (root / "split_manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


class HoldoutValidationTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self):
        self._folder.cleanup()

    def five_cases(self, **options) -> list[Path]:
        return [make_case(self.root / f"m{index}", index=index, **options)
                for index in range(1, MIN_HOLDOUT_MATCHES + 1)]

    def frozen_list(self, paths: list[Path]) -> Path:
        list_path = self.root / "holdout.json"
        list_path.write_text(json.dumps({"cases": [
            {"path": str(path.relative_to(self.root)), "sha256": sha256(path)} for path in paths]}),
            encoding="utf-8")
        return list_path

    # End-to-end synthetic fixture.

    def test_synthetic_holdout_end_to_end_through_command_line(self):
        paths = self.five_cases()
        list_path = self.frozen_list(paths)
        dataset = make_dataset(self.root / "dataset")
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        subprocess.run([sys.executable, "-m", "control_vision.validation.holdout",
                        "--holdout-list", list_path.name, "--dataset-root", "dataset",
                        "--output", "report.json"],
                       cwd=self.root, env=environment, check=True, capture_output=True, text=True, timeout=60)
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        summary = report["summary"]
        self.assertEqual(summary["match_count"], 5)
        self.assertEqual(set(summary["phases"].values()), {"pass"})
        self.assertTrue(summary["training_disjointness"]["verified"])
        self.assertTrue(summary["holdout_list"]["verified"])
        self.assertEqual(summary["performance"]["status"], "not_measured")
        self.assertTrue(summary["acceptance_targets_met"], summary["blockers"])
        self.assertEqual(summary["tracking"]["visible_observations"], 5 * 20 * 2)
        self.assertEqual(len(summary["per_match"]), 5)
        case = report["cases"][0]
        self.assertEqual(case["inputs"]["case"]["sha256"], sha256(paths[0]))
        self.assertEqual(case["inputs"]["project_db"]["sha256"], sha256(paths[0].parent / "project.sqlite"))
        self.assertEqual(case["inputs"]["video"]["sha256"], sha256(paths[0].parent / "match.mp4"))
        self.assertEqual(case["inputs"]["clips"][0]["sha256"], sha256(paths[0].parent / "clip.mp4"))

    def test_without_frozen_list_or_training_manifest_targets_are_not_met(self):
        report = aggregate([evaluate_case(path) for path in self.five_cases()])
        self.assertEqual(set(report["summary"]["phases"].values()), {"pass"})
        self.assertFalse(report["summary"]["acceptance_targets_met"])
        self.assertIn("holdout/training disjointness not verified", report["summary"]["blockers"])
        self.assertIn("holdout set not frozen by a hashed list", report["summary"]["blockers"])

    def test_command_line_template_from_another_directory(self):
        make_case(self.root)
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        subprocess.run([sys.executable, "-m", "control_vision.validation.holdout", "--make-template",
                        "--project", "project.sqlite", "--video-id", "1", "--sample-fps", "2",
                        "--output", "template.json"],
                       cwd=self.root, env=environment, check=True, capture_output=True, text=True, timeout=60)
        template = json.loads((self.root / "template.json").read_text(encoding="utf-8"))
        self.assertEqual(len(template["tracks"]), 40)
        self.assertIsNone(template["tracks"][0]["visible"])
        self.assertEqual(template["format_version"], 1)
        self.assertIsNone(template["official_scores"]["source_url"])

    def test_output_never_overwrites_labels_without_force(self):
        case_path = make_case(self.root)
        labels = case_path.read_text(encoding="utf-8")
        template = ["--make-template", "--project", str(self.root / "project.sqlite"), "--video-id", "1",
                    "--sample-fps", "2", "--output", str(case_path)]
        with self.assertRaises(SystemExit):
            main(template)
        self.assertEqual(case_path.read_text(encoding="utf-8"), labels)
        with self.assertRaises(SystemExit):
            main([str(case_path), "--output", str(case_path), "--force"])
        self.assertEqual(case_path.read_text(encoding="utf-8"), labels)
        report_path = self.root / "report.json"
        main([str(case_path), "--output", str(report_path)])
        first = report_path.read_text(encoding="utf-8")
        with self.assertRaises(SystemExit):
            main([str(case_path), "--output", str(report_path)])
        main(template[:-1] + [str(report_path), "--force"])
        self.assertNotEqual(report_path.read_text(encoding="utf-8"), first)
        self.assertEqual([path.name for path in self.root.glob(".*.tmp")], [])

    def test_force_never_replaces_run_inputs(self):
        paths = self.five_cases()
        list_path = self.frozen_list(paths)
        frozen = list_path.read_bytes()
        db_path = paths[0].parent / "project.sqlite"
        database = db_path.read_bytes()
        dataset = make_dataset(self.root / "dataset")
        for target in (list_path, db_path, paths[0].parent / "match.mp4", paths[0].parent / "clip.mp4",
                       dataset / "split_manifest.csv"):
            with self.subTest(target=target.name), self.assertRaises(SystemExit):
                main(["--holdout-list", str(list_path), "--dataset-root", str(dataset),
                      "--output", str(target), "--force"])
        with self.assertRaises(SystemExit):
            main(["--make-template", "--project", str(db_path), "--video-id", "1", "--sample-fps", "2",
                  "--output", str(db_path), "--force"])
        self.assertEqual(list_path.read_bytes(), frozen)
        self.assertEqual(db_path.read_bytes(), database)

    def test_failed_fallback_write_leaves_no_partial_output(self):
        target = self.root / "report.json"
        real_open = Path.open

        class FailingStream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.stream.close()

            def write(self, text):
                self.stream.write(text[:3])
                raise OSError("disk full")

        def fake_open(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            return FailingStream(stream) if mode == "x" else stream

        with mock.patch("os.link", side_effect=OSError("no hard links")), \
                mock.patch.object(Path, "open", fake_open), self.assertRaisesRegex(OSError, "disk full"):
            write_output(target, "{\"report\": 1}\n")
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        target.write_text("someone else's file", encoding="utf-8")
        with mock.patch("os.link", side_effect=OSError("no hard links")), self.assertRaises(FileExistsError):
            write_output(target, "{}\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "someone else's file")

    def test_database_path_with_url_characters_opens_the_right_file(self):
        case = evaluate_case(make_case(self.root / "Qual #3 100%"))
        self.assertEqual(case["tracking"]["completeness"], 1)

    # Minimum match count and "not measured".

    def test_single_match_never_reports_a_pass(self):
        report = aggregate([evaluate_case(make_case(self.root))])
        self.assertEqual(set(report["summary"]["phases"].values()), {"insufficient_matches"})
        self.assertFalse(report["summary"]["acceptance_targets_met"])

    def test_four_matches_are_insufficient_for_cycles(self):
        cases = [evaluate_case(path) for path in self.five_cases()[:4]]
        self.assertEqual(aggregate(cases)["summary"]["phases"]["phase_2_cycles"], "insufficient_matches")

    def test_no_scoring_evidence_is_not_measured_rather_than_pass_or_fail(self):
        cases = []
        for path in self.five_cases():
            db = ProjectDatabase(path.parent / "project.sqlite")
            db.connection.execute("DELETE FROM scoring_events")
            db.connection.commit()
            db.close()
            edit_truth(path, lambda truth: truth.update(scoring_events=[]))
            cases.append(evaluate_case(path))
        summary = aggregate(cases)["summary"]
        self.assertEqual(summary["phases"]["phase_3_highlights"], "not_measured")
        self.assertIsNone(summary["highlights"]["clip_coverage"])
        self.assertIsNone(summary["scoring"]["robot_attribution_fraction"])
        self.assertFalse(summary["acceptance_targets_met"])

    # Tracking metric definitions.

    def test_identity_switches_are_counted_even_when_tracks_are_present(self):
        frames = range(0, SECONDS * FPS, STRIDE)
        tracks = {frame: (((2.0, 2.0), (10.0, 10.0)) if frame < 20 else ((10.0, 10.0), (2.0, 2.0)))
                  for frame in frames}
        case = evaluate_case(make_case(self.root, tracks=tracks))
        self.assertEqual(case["tracking"]["completeness"], 1)
        self.assertEqual(case["tracking"]["identity_correct_fraction"], .5)
        self.assertEqual(case["tracking"]["id_switches"], 2)

    def test_close_robots_keep_their_track_while_inside_the_gate(self):
        # Truth robots 1 ft apart; predictions drift so a fresh min-distance
        # assignment would swap them, but both stay within the 1.5 ft gate.
        frames = list(range(0, SECONDS * FPS, STRIDE))
        truth = {frame: ((2.0, 2.0), (3.0, 2.0)) for frame in frames}
        tracks = {frame: (((2.0, 2.0), (3.0, 2.0)) if frame == 0 else ((2.6, 2.0), (2.4, 2.0)))
                  for frame in frames}
        case = evaluate_case(make_case(self.root, tracks=tracks, truth_tracks=truth))
        self.assertEqual(case["tracking"]["id_switches"], 0)
        self.assertEqual(case["tracking"]["identity_aware_completeness"], 1)

    def test_fragmentation_is_not_an_identity_switch(self):
        truth = {frame: [{"team_number": RED, "x_ft": 2.0, "y_ft": 2.0}] for frame in range(6)}
        predicted = {0: [{"team_number": RED, "x_ft": 2.0, "y_ft": 2.0}],
                     3: [{"team_number": RED, "x_ft": 2.1, "y_ft": 2.0}],
                     5: [{"team_number": BLUE, "x_ft": 2.0, "y_ft": 2.0}]}
        result = match_tracks(truth, predicted)
        self.assertEqual(result["matched"][RED], 3)
        self.assertEqual(result["switches"][RED], 1)

    def test_hungarian_assignment_matches_exhaustive_search(self):
        rng = random.Random(7)
        for _ in range(200):
            rows, columns = rng.randint(1, 5), rng.randint(1, 5)
            cost = [[rng.uniform(0, 3) for _ in range(columns)] for _ in range(rows)]
            pairs = assign(cost, 1.5)
            best = (0, 0.0)
            for size in range(min(rows, columns), 0, -1):
                for chosen in itertools.combinations(range(rows), size):
                    for perm in itertools.permutations(range(columns), size):
                        values = [cost[r][c] for r, c in zip(chosen, perm)]
                        if all(value <= 1.5 for value in values):
                            candidate = (size, -sum(values))
                            best = max(best, candidate)
            self.assertEqual(len(pairs), best[0])
            if pairs:
                self.assertTrue(math.isclose(sum(p[2] for p in pairs), -best[1], abs_tol=1e-9))
            self.assertEqual(len({p[0] for p in pairs}), len(pairs))
            self.assertEqual(len({p[1] for p in pairs}), len(pairs))

    def test_consistently_swapped_robots_cannot_pass_tracking_target(self):
        swapped = {frame: ((10.0, 10.0), (2.0, 2.0)) for frame in range(0, SECONDS * FPS, STRIDE)}
        cases = [evaluate_case(make_case(self.root / f"m{i}", index=i, tracks=swapped)) for i in range(1, 6)]
        self.assertEqual(cases[0]["tracking"]["completeness"], 1)
        self.assertEqual(cases[0]["tracking"]["id_switches"], 0)
        self.assertEqual(cases[0]["tracking"]["identity_aware_completeness"], 0)
        self.assertEqual(aggregate(cases)["summary"]["phases"]["phase_1_tracking"], "fail")

    def test_fully_occluded_seeded_robot_fails_metric_without_crashing(self):
        hidden = {frame: ((2.0, 2.0), None) for frame in range(0, SECONDS * FPS, STRIDE)}
        case = evaluate_case(make_case(self.root, truth_tracks=hidden))
        self.assertIsNone(case["tracking"]["per_robot"][str(BLUE)]["identity_aware_completeness"])
        summary = aggregate([case])["summary"]
        self.assertEqual(summary["tracking"]["minimum_per_robot_identity_aware_completeness"], 0)
        self.assertEqual(summary["phases"]["phase_1_tracking"], "fail")

    def test_pixel_box_truth_is_projected_like_the_app(self):
        case_path = make_case(self.root)

        def to_boxes(truth):
            for row in truth["tracks"]:
                x, y = row.pop("x_ft"), row.pop("y_ft")
                # A 3 ft tall box: projecting its top edge would miss the gate.
                row["box_px"] = [x * 10 - 5, y * 10 - 30, x * 10 + 5, y * 10]
        edit_truth(case_path, to_boxes)
        case = evaluate_case(case_path)
        self.assertEqual(case["tracking"]["completeness"], 1)
        self.assertEqual(case["tracking"]["identity_aware_completeness"], 1)

    # Highlights.

    def test_overlong_clip_does_not_meet_the_three_second_window(self):
        case = evaluate_case(make_case(self.root, clip_window=(0.0, 10.0)))
        self.assertEqual(case["highlights"]["active_events_with_clip"], 1)
        self.assertEqual(case["highlights"]["active_events_with_correct_window"], 0)
        # Saved bounds alone must match too, even when the file is 6 s long.
        saved = evaluate_case(make_case(self.root / "saved", clip_window=(0.0, 10.0), clip_seconds=6.0))
        self.assertEqual(saved["highlights"]["active_events_with_correct_window"], 0)

    def test_clip_file_shorter_than_its_saved_window_fails(self):
        case = evaluate_case(make_case(self.root, clip_seconds=2.0))
        self.assertEqual(case["highlights"]["active_events_with_correct_window"], 0)

    def test_clip_window_is_clamped_at_video_bounds(self):
        case = evaluate_case(make_case(self.root, event_t=1.0, clip_window=(0.0, 4.0)))
        self.assertEqual(case["highlights"]["active_events_with_correct_window"], 1)
        late = evaluate_case(make_case(self.root / "late", event_t=9.0, clip_window=(6.0, 10.0)))
        self.assertEqual(late["highlights"]["active_events_with_correct_window"], 1)

    def test_missing_truth_scoring_event_fails_clip_coverage(self):
        cases = [evaluate_case(path) for path in self.five_cases(
            extra_truth_events=[{"t_s": 7.5, "team_number": BLUE, "event_type": "flower_element"}])]
        self.assertEqual(cases[0]["highlights"]["active_events_with_correct_window"], 1)
        self.assertEqual(cases[0]["highlights"]["truth_events_with_correct_clip"], 1)
        self.assertEqual(aggregate(cases)["summary"]["phases"]["phase_3_highlights"], "fail")

    def test_relative_clip_path_resolves_from_database_directory(self):
        case_path = make_case(self.root)
        db = ProjectDatabase(self.root / "project.sqlite")
        db.connection.execute("UPDATE clips SET path=?", ("clip.mp4",))
        db.connection.commit()
        db.close()
        self.assertEqual(evaluate_case(case_path)["highlights"]["active_events_with_correct_window"], 1)

    # Scoring.

    def test_offsetting_alliance_errors_do_not_pass_the_score_check(self):
        case = evaluate_case(make_case(self.root, official=(0, 2)))
        self.assertFalse(case["scoring"]["within_ten_percent"])

    def test_missed_human_events_lower_robot_attribution(self):
        extra = [{"t_s": t, "team_number": BLUE, "event_type": "flower_element"} for t in (1.0, 2.0, 3.0, 4.0)]
        case = evaluate_case(make_case(self.root, extra_truth_events=extra))
        self.assertEqual(case["scoring"]["attribution_denominator"], 5)
        self.assertEqual(case["scoring"]["attribution_fraction"], .2)

    def test_app_events_without_human_event_lower_robot_attribution(self):
        case_path = make_case(self.root)
        db = ProjectDatabase(self.root / "project.sqlite")
        robot = db.robots_for_video(1)[0]["id"]
        extra = db.add_scoring_event(robot, 8.0)
        db.review_event(extra, "confirmed", 2, robot, "flower_element", "high", "red")
        db.close()
        case = evaluate_case(case_path)
        self.assertEqual(case["scoring"]["unmatched_app_scored_events"], 1)
        self.assertEqual(case["scoring"]["attribution_fraction"], .5)

    def test_official_scores_need_an_official_source(self):
        case_path = make_case(self.root)
        edit_truth(case_path, lambda truth: truth["official_scores"].pop("source_url"))
        with self.assertRaisesRegex(ValueError, "source_url"):
            evaluate_case(case_path)
        edit_truth(case_path, lambda truth: truth["official_scores"].update(
            source_url="https://example.com/results"))
        with self.assertRaisesRegex(ValueError, "ftc-events.firstinspires.org"):
            evaluate_case(case_path)

    def test_stored_official_score_must_agree_with_truth(self):
        case_path = make_case(self.root)
        db = ProjectDatabase(self.root / "project.sqlite")
        db.set_official_scores(1, 30, 0)
        db.close()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            evaluate_case(case_path)

    # Holdout set integrity.

    def test_holdout_match_from_a_training_event_is_rejected(self):
        cases = [evaluate_case(make_case(self.root / "m1"))]
        dataset = make_dataset(self.root / "dataset", train_event="USMOSTLN")
        with self.assertRaisesRegex(ValueError, "events used for training"):
            check_training_disjoint(cases, dataset)
        clean = check_training_disjoint(cases, make_dataset(self.root / "clean"))
        self.assertTrue(clean["verified"])

    def test_changed_case_after_freezing_is_rejected(self):
        paths = self.five_cases()
        list_path = self.frozen_list(paths)
        self.assertEqual(len(load_holdout_list(list_path)[0]), 5)
        edit_truth(paths[2], lambda truth: truth["cycles"][0].update(count=1))
        with self.assertRaisesRegex(ValueError, "changed after the list was frozen"):
            load_holdout_list(list_path)

    def test_same_match_or_video_cannot_be_counted_twice(self):
        case = evaluate_case(make_case(self.root))
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate([case, case])
        other = evaluate_case(make_case(self.root / "copy"))
        other["match_label"] = "different label"
        with self.assertRaisesRegex(ValueError, "only once"):
            aggregate([case, other])

    def test_preview_mode_cannot_meet_targets(self):
        paths = self.five_cases(tracking_mode="seeded_optical_flow")
        report = aggregate([evaluate_case(path) for path in paths])
        self.assertTrue(report["summary"]["preview_mode_used"])
        self.assertFalse(report["summary"]["acceptance_targets_met"])

    # Truth validation.

    def test_negative_hand_counted_cycles_are_rejected(self):
        case_path = make_case(self.root)
        edit_truth(case_path, lambda truth: truth["cycles"][0].update(count=-1))
        with self.assertRaisesRegex(ValueError, "nonnegative whole numbers"):
            evaluate_case(case_path)

    def test_failed_rerun_cannot_reuse_stale_saved_tracks(self):
        case_path = make_case(self.root)
        db = ProjectDatabase(self.root / "project.sqlite")
        db.set_analysis_status(1, "failed")
        db.close()
        with self.assertRaisesRegex(ValueError, "failed analysis"):
            evaluate_case(case_path)

    def test_incomplete_truth_cannot_claim_coverage(self):
        case_path = make_case(self.root)
        edit_truth(case_path, lambda truth: truth["tracks"].pop())
        with self.assertRaisesRegex(ValueError, "every seeded team"):
            evaluate_case(case_path)

    def test_template_requires_human_visibility_labels(self):
        case_path = make_case(self.root)
        truth = json.loads(case_path.read_text(encoding="utf-8"))
        template = make_template(self.root / "project.sqlite", 1, 2.0)
        self.assertEqual(len(template["tracks"]), 40)
        template.update(match_label="synthetic Q1", source_match="USMOSTLN:Q1",
                        calibration_landmarks=truth["calibration_landmarks"],
                        official_scores=truth["official_scores"],
                        cycles=truth["cycles"])
        case_path.write_text(json.dumps(template), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reviewed true/false"):
            evaluate_case(case_path)

    def test_duplicate_landmarks_cannot_fake_reprojection_evidence(self):
        case_path = make_case(self.root)
        edit_truth(case_path, lambda truth: truth.update(
            calibration_landmarks=[truth["calibration_landmarks"][0]] * 4))
        with self.assertRaisesRegex(ValueError, "distinct field positions"):
            evaluate_case(case_path)

    def test_collinear_landmarks_cannot_fake_reprojection_evidence(self):
        case_path = make_case(self.root)
        edit_truth(case_path, lambda truth: truth.update(calibration_landmarks=[
            {"x_ft": x, "y_ft": x, "pixel_x": x * 10, "pixel_y": x * 10} for x in (2, 4, 6, 8)]))
        with self.assertRaisesRegex(ValueError, "span an area"):
            evaluate_case(case_path)

    def test_fitted_corner_cannot_count_as_independent_landmark(self):
        case_path = make_case(self.root)
        edit_truth(case_path, lambda truth: truth["calibration_landmarks"].__setitem__(
            0, {"x_ft": 0, "y_ft": 0, "pixel_x": 0, "pixel_y": 0}))
        with self.assertRaisesRegex(ValueError, "independent of the fitted corners"):
            evaluate_case(case_path)

    def test_reprojection_error_uses_held_out_landmarks(self):
        case_path = make_case(self.root)
        edit_truth(case_path, lambda truth: truth["calibration_landmarks"][0].update(pixel_x=36))
        case = evaluate_case(case_path)
        self.assertAlmostEqual(case["calibration"]["max_reprojection_error_px"], 6, places=3)


if __name__ == "__main__":
    unittest.main()
