import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from control_vision.storage.database import ProjectDatabase
from control_vision.validation.holdout import aggregate, evaluate_case, make_template


class HoldoutValidationTests(unittest.TestCase):
    def make_case(self, folder: Path):
        video_path = folder / "match.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (160, 160))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(np.zeros((160, 160, 3), dtype=np.uint8))
        writer.release()
        db = ProjectDatabase(folder / "project.sqlite")
        project_id = db.add_project("holdout")
        video_id = db.add_video(project_id, video_path,
                                {"duration_s": 1.0, "fps": 4, "width": 160, "height": 160})
        db.save_calibration(video_id, [[0, 0], [100, 0], [100, 100], [0, 100]])
        red = db.add_robot(video_id, 1234, "red", 20, 20)
        blue = db.add_robot(video_id, 5678, "blue", 80, 80)
        db.replace_tracks([(red, 0, 0.0, 2, 2, .9), (blue, 0, 0.0, 10, 10, .9),
                           (red, 2, 0.5, 2.2, 2, .9), (blue, 2, 0.5, 10.2, 10, .9)])
        event_id = db.add_scoring_event(red, 0.5)
        db.review_event(event_id, "confirmed", 2, red, "flower_element", "high", "red")
        clip_path = folder / "clip.mp4"
        clip_path.write_bytes(video_path.read_bytes())
        db.add_clip(event_id, clip_path, 0.0, 1.0)
        db.close()
        truth = {
            "match_label": "synthetic Q1", "project_db": "project.sqlite", "video_id": video_id,
            "sample_fps": 2.0,
            "tracks": [
                {"frame": frame, "team_number": team, "visible": True,
                 "x_ft": x, "y_ft": y}
                for frame, positions in ((0, ((1234, 2, 2), (5678, 10, 10))),
                                         (2, ((1234, 2.2, 2), (5678, 10.2, 10))))
                for team, x, y in positions
            ],
            "cycles": [{"team_number": 1234, "count": 0},
                       {"team_number": 5678, "count": 0}],
            "scoring_events": [{"t_s": .5, "team_number": 1234,
                                "event_type": "flower_element"}],
            "official_scores": {"red": 2, "blue": 0},
            "calibration_landmarks": [
                {"x_ft": x, "y_ft": y, "pixel_x": x / 12 * 100,
                 "pixel_y": y / 12 * 100}
                for x, y in ((3, 3), (9, 3), (3, 9), (9, 9))
            ],
        }
        case_path = folder / "case.json"
        case_path.write_text(json.dumps(truth), encoding="utf-8")
        return case_path, db, red, blue

    def test_reports_complete_synthetic_case_and_requires_five_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            case = evaluate_case(case_path)
            self.assertEqual(case["tracking"]["completeness"], 1)
            self.assertEqual(case["tracking"]["id_switches"], 0)
            self.assertEqual(case["highlights"]["clips_cover_three_seconds_each_side"], 1)
            self.assertTrue(case["scoring"]["within_ten_percent"])
            self.assertEqual(case["scoring"]["attribution_fraction"], 1)
            self.assertFalse(aggregate([case])["summary"]["brief_targets_met"])
            with self.assertRaisesRegex(ValueError, "only once"):
                aggregate([case] * 5)
            distinct = []
            for index in range(5):
                report = deepcopy(case)
                report["match_label"] = f"synthetic Q{index + 1}"
                report["video_id"] = index + 1
                report["video_path"] = f"synthetic-Q{index + 1}.mp4"
                distinct.append(report)
            self.assertTrue(aggregate(distinct)["summary"]["brief_targets_met"])
            distinct[0]["tracking_mode"] = "seeded_optical_flow"
            self.assertFalse(aggregate(distinct)["summary"]["brief_targets_met"])

    def test_identity_switches_are_counted_even_when_tracks_are_present(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, red, blue = self.make_case(Path(folder))
            db = ProjectDatabase(Path(folder) / "project.sqlite")
            db.replace_tracks([(red, 0, 0.0, 2, 2, .9), (blue, 0, 0.0, 10, 10, .9),
                               (red, 2, 0.5, 10.2, 10, .9), (blue, 2, 0.5, 2.2, 2, .9)])
            db.close()
            case = evaluate_case(case_path)
            self.assertEqual(case["tracking"]["completeness"], 1)
            self.assertEqual(case["tracking"]["identity_correct_fraction"], .5)
            self.assertEqual(case["tracking"]["id_switches"], 2)

    def test_consistently_swapped_robots_cannot_pass_tracking_target(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, red, blue = self.make_case(Path(folder))
            db = ProjectDatabase(Path(folder) / "project.sqlite")
            db.replace_tracks([(red, 0, 0.0, 10, 10, .9), (blue, 0, 0.0, 2, 2, .9),
                               (red, 2, 0.5, 10.2, 10, .9), (blue, 2, 0.5, 2.2, 2, .9)])
            db.close()
            case = evaluate_case(case_path)
            self.assertEqual(case["tracking"]["completeness"], 1)
            self.assertEqual(case["tracking"]["id_switches"], 0)
            self.assertEqual(case["tracking"]["identity_aware_completeness"], 0)
            for robot in case["tracking"]["per_robot"].values():
                self.assertEqual(robot["identity_aware_completeness"], 0)
            distinct = []
            for index in range(5):
                report = deepcopy(case)
                report["match_label"] = f"swapped Q{index + 1}"
                report["video_id"] = index + 1
                report["video_path"] = f"swapped-Q{index + 1}.mp4"
                distinct.append(report)
            self.assertFalse(aggregate(distinct)["summary"]["phase_1_pass"])

    def test_fully_occluded_seeded_robot_fails_metric_without_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            truth = json.loads(case_path.read_text(encoding="utf-8"))
            for row in truth["tracks"]:
                if row["team_number"] == 5678:
                    row.update(visible=False, x_ft=None, y_ft=None)
            case_path.write_text(json.dumps(truth), encoding="utf-8")
            case = evaluate_case(case_path)
            self.assertIsNone(case["tracking"]["per_robot"]["5678"]["identity_aware_completeness"])
            report = aggregate([case])
            self.assertEqual(report["summary"]["minimum_per_robot_identity_aware_completeness"], 0)
            self.assertFalse(report["summary"]["phase_1_pass"])

    def test_missing_truth_scoring_event_fails_clip_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            truth = json.loads(case_path.read_text(encoding="utf-8"))
            truth["scoring_events"].append({
                "t_s": 0.75, "team_number": 5678, "event_type": "flower_element"})
            case_path.write_text(json.dumps(truth), encoding="utf-8")
            case = evaluate_case(case_path)
            self.assertEqual(case["highlights"]["clips_cover_three_seconds_each_side"], 1)
            self.assertEqual(case["highlights"]["truth_events"], 2)
            self.assertEqual(case["highlights"]["truth_clips_cover_three_seconds_each_side"], 1)
            self.assertFalse(aggregate([case])["summary"]["phase_3_pass"])

    def test_negative_hand_counted_cycles_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            truth = json.loads(case_path.read_text(encoding="utf-8"))
            truth["cycles"][0]["count"] = -1
            case_path.write_text(json.dumps(truth), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonnegative whole numbers"):
                evaluate_case(case_path)

    def test_incomplete_truth_cannot_claim_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            truth = json.loads(case_path.read_text(encoding="utf-8"))
            truth["tracks"].pop()
            case_path.write_text(json.dumps(truth), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every seeded team"):
                evaluate_case(case_path)

    def test_template_requires_human_visibility_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            truth = json.loads(case_path.read_text(encoding="utf-8"))
            template = make_template(Path(folder) / "project.sqlite", truth["video_id"], 2.0)
            self.assertEqual(len(template["tracks"]), 4)
            self.assertIsNone(template["tracks"][0]["visible"])
            template["match_label"] = "synthetic Q1"
            template["calibration_landmarks"] = truth["calibration_landmarks"]
            case_path.write_text(json.dumps(template), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reviewed true/false"):
                evaluate_case(case_path)

    def test_duplicate_landmarks_cannot_fake_reprojection_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            truth = json.loads(case_path.read_text(encoding="utf-8"))
            truth["calibration_landmarks"] = [truth["calibration_landmarks"][0]] * 4
            case_path.write_text(json.dumps(truth), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distinct field positions"):
                evaluate_case(case_path)

    def test_relative_clip_path_resolves_from_database_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            db = ProjectDatabase(Path(folder) / "project.sqlite")
            db.connection.execute("UPDATE clips SET path=?", ("clip.mp4",))
            db.connection.commit()
            db.close()
            case = evaluate_case(case_path)
            self.assertEqual(case["highlights"]["clips_present"], 1)
            self.assertEqual(case["highlights"]["clips_cover_three_seconds_each_side"], 1)

    def test_collinear_landmarks_cannot_fake_reprojection_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            truth = json.loads(case_path.read_text(encoding="utf-8"))
            truth["calibration_landmarks"] = [
                {"x_ft": x, "y_ft": x, "pixel_x": x / 12 * 100,
                 "pixel_y": x / 12 * 100} for x in (2, 4, 6, 8)]
            case_path.write_text(json.dumps(truth), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "span an area"):
                evaluate_case(case_path)

    def test_fitted_corner_cannot_count_as_independent_landmark(self):
        with tempfile.TemporaryDirectory() as folder:
            case_path, _, _, _ = self.make_case(Path(folder))
            truth = json.loads(case_path.read_text(encoding="utf-8"))
            truth["calibration_landmarks"][0] = {
                "x_ft": 0, "y_ft": 0, "pixel_x": 0, "pixel_y": 0}
            case_path.write_text(json.dumps(truth), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent of the fitted corners"):
                evaluate_case(case_path)


if __name__ == "__main__":
    unittest.main()
