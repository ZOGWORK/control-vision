import unittest
import sqlite3
import tempfile
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from control_vision.pipeline.scoreboard_ocr import _consensus, _read_digits
from control_vision.pipeline.tracking import (
    _match_initial_seeds, _new_roi_candidate, _nearest_robot_id,
    _unmatched_roi_detections,
)
from control_vision.pipeline.tracking import track_video
from control_vision.pipeline.ingest import probe_video
from control_vision.pipeline.manual_tracking import _photo_crop_bounds, _save_best_photos
from control_vision.pipeline.zones import zone_at
from control_vision.game.scoreboard import calculate_scoreboard
from control_vision.storage.database import ProjectDatabase, SCHEMA
from control_vision.app.main_window import ProcessingWorker


class WorkerDispatchTests(unittest.TestCase):
    def test_seeded_mode_passes_progress_logger_and_photos_in_order(self):
        logger = logging.getLogger("control-vision-worker-test")
        photos_dir = Path("photos")
        worker = ProcessingWorker((Path("match.mp4"), Path("unused.onnx"), [], [(1, 20, 30)],
                                   8.0, logger, photos_dir, 0.25, None, "",
                                   "seeded_optical_flow", {1: "red"}))
        with patch("control_vision.app.main_window.track_seeded_video", autospec=True,
                   return_value=([], [], [])) as tracker:
            worker.run()
        args = tracker.call_args.args
        self.assertEqual(args[:5], (Path("match.mp4"), [], [(1, 20, 30)], {1: "red"}, 8.0))
        self.assertTrue(callable(args[5]))
        self.assertTrue(callable(args[6]))
        self.assertIs(args[7], logger)
        self.assertEqual(args[8], photos_dir)


class SeedMatchingTests(unittest.TestCase):
    def test_failed_model_initialization_releases_video(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop):
                import cv2
                return 30 if prop == cv2.CAP_PROP_FPS else 90
            def release(self): self.released = True

        capture = Capture()
        def fail_model(*args, **kwargs):
            raise RuntimeError("bad model")
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "robot.onnx"
            model_path.write_bytes(b"placeholder")
            with patch("control_vision.pipeline.tracking.cv2.VideoCapture", return_value=capture), \
                 patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=fail_model)}):
                with self.assertRaisesRegex(RuntimeError, "bad model"):
                    track_video(Path("match.mp4"), model_path, [[0, 0], [1, 0], [1, 1], [0, 1]],
                                [(1, 0, 0)], 8.0, lambda: False, lambda *_: None,
                                logging.getLogger("model-init-test"))
        self.assertTrue(capture.released)

    def test_repeated_model_inference_errors_fail_before_replacing_results(self):
        import numpy as np
        from types import SimpleNamespace

        class Capture:
            reads = 0
            def isOpened(self): return True
            def get(self, prop):
                import cv2
                return 30 if prop == cv2.CAP_PROP_FPS else 35
            def read(self):
                self.reads += 1
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            def release(self): pass

        def fail(*args, **kwargs):
            raise RuntimeError("incompatible runtime")

        capture = Capture()
        model = SimpleNamespace(track=fail)
        logger = logging.getLogger("model-failure-test")
        logger.propagate = False
        logger.addHandler(logging.NullHandler())
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "robot.onnx"
            model_path.touch()
            with patch("control_vision.pipeline.tracking.cv2.VideoCapture", return_value=capture), \
                 patch("ultralytics.YOLO", return_value=model):
                with self.assertRaisesRegex(RuntimeError, "three consecutive sampled frames"):
                    track_video(Path("match.mp4"), model_path,
                                [[0, 0], [100, 0], [100, 100], [0, 100]], [(1, 90, 90)],
                                30.0, lambda: False, lambda *_: None, logger)
        self.assertEqual(capture.reads, 3)

    def test_one_inference_error_does_not_discard_later_robot_track(self):
        import numpy as np
        from types import SimpleNamespace

        class Capture:
            reads = 0
            def isOpened(self): return True
            def get(self, prop):
                import cv2
                return 30 if prop == cv2.CAP_PROP_FPS else 2
            def read(self):
                self.reads += 1
                return (True, np.zeros((480, 640, 3), dtype=np.uint8)) if self.reads <= 2 else (False, None)
            def release(self): pass

        box = SimpleNamespace(cls=[0], xyxy=np.array([[40, 40, 60, 90]]), conf=[.9], id=[1])
        results = iter([RuntimeError("one bad frame"), [SimpleNamespace(boxes=[box])]])
        def detect(*args, **kwargs):
            result = next(results)
            if isinstance(result, Exception): raise result
            return result
        logger = logging.getLogger("single-frame-failure-test")
        logger.propagate = False
        logger.addHandler(logging.NullHandler())
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "robot.onnx"
            model_path.touch()
            with patch("control_vision.pipeline.tracking.cv2.VideoCapture", return_value=Capture()), \
                 patch("ultralytics.YOLO", return_value=SimpleNamespace(track=detect)):
                rows, _, _ = track_video(Path("match.mp4"), model_path,
                                         [[0, 0], [100, 0], [100, 100], [0, 100]], [(1, 50, 90)],
                                         30.0, lambda: False, lambda *_: None, logger)
        self.assertEqual(len(rows), 1)

    def test_yolo_stops_after_thirty_empty_sampled_frames(self):
        import numpy as np
        from types import SimpleNamespace

        class Capture:
            reads = 0
            def isOpened(self): return True
            def get(self, prop):
                import cv2
                return 30 if prop == cv2.CAP_PROP_FPS else 35
            def read(self):
                self.reads += 1
                return (True, np.zeros((480, 640, 3), dtype=np.uint8)) if self.reads <= 35 else (False, None)
            def release(self): pass

        capture = Capture()
        model = SimpleNamespace(track=lambda *args, **kwargs: [SimpleNamespace(boxes=[])])
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "robot.onnx"
            model_path.touch()
            with patch("control_vision.pipeline.tracking.cv2.VideoCapture", return_value=capture), \
                 patch("ultralytics.YOLO", return_value=model):
                with self.assertRaisesRegex(RuntimeError, "No robots were detected in the first 30"):
                    track_video(Path("match.mp4"), model_path,
                                [[0, 0], [100, 0], [100, 100], [0, 100]], [(1, 90, 90)],
                                30.0, lambda: False, lambda *_: None,
                                logging.getLogger("empty-model-test"))
        self.assertEqual(capture.reads, 30)

    def test_goal_candidate_does_not_guess_a_missing_or_distant_robot(self):
        self.assertIsNone(_nearest_robot_id([], 100, 100, 150))
        self.assertIsNone(_nearest_robot_id([(7, 400, 400)], 100, 100, 150))
        self.assertEqual(_nearest_robot_id([(7, 120, 120)], 100, 100, 150), 7)

    def test_yolo_track_outside_calibrated_field_is_not_saved(self):
        import numpy as np
        from types import SimpleNamespace

        class Capture:
            reads = 0
            def isOpened(self): return True
            def get(self, prop):
                import cv2
                return 30 if prop == cv2.CAP_PROP_FPS else 1
            def read(self):
                self.reads += 1
                return (True, np.zeros((480, 640, 3), dtype=np.uint8)) if self.reads == 1 else (False, None)
            def release(self): pass

        box = SimpleNamespace(cls=[0], xyxy=np.array([[140, 100, 160, 150]]),
                              conf=[0.9], id=[1])
        model = SimpleNamespace(track=lambda *args, **kwargs: [SimpleNamespace(boxes=[box])])
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "robot.onnx"
            model_path.touch()
            with patch("control_vision.pipeline.tracking.cv2.VideoCapture", return_value=Capture()), \
                 patch("ultralytics.YOLO", return_value=model):
                with self.assertRaisesRegex(RuntimeError, "No usable robot tracks"):
                    track_video(Path("match.mp4"), model_path,
                                [[0, 0], [100, 0], [100, 100], [0, 100]], [(1, 90, 90)],
                                8.0, lambda: False, lambda *_: None,
                                logging.getLogger("field-test"))

    def test_goal_element_reacquisition_does_not_repeat_candidate(self):
        recent = [(100.0, 100.0, 1.0)]
        self.assertFalse(_new_roi_candidate(105.0, 99.0, 2.0, recent))
        self.assertTrue(_new_roi_candidate(180.0, 100.0, 2.0, recent))
        self.assertTrue(_new_roi_candidate(100.0, 100.0, 4.0, recent))

    def test_new_goal_element_is_found_when_detector_reorders_roi_detections(self):
        # Previously known A shifts slightly; newly arrived B is listed first.
        known = [(100.0, 100.0)]
        current = [(200.0, 100.0, 0.91), (102.0, 101.0, 0.88)]
        self.assertEqual(_unmatched_roi_detections(current, known), [current[0]])

    def test_new_goal_element_is_found_when_roi_count_stays_the_same(self):
        # A can leave as B enters. Count deltas alone miss this replacement.
        known = [(100.0, 100.0)]
        current = [(180.0, 100.0, 0.93)]
        self.assertEqual(_unmatched_roi_detections(current, known), current)

    def test_track_video_flags_new_roi_element_after_detector_reorders_list(self):
        import numpy as np
        from types import SimpleNamespace

        class Capture:
            reads = 0
            def isOpened(self): return True
            def get(self, prop):
                import cv2
                return 30 if prop == cv2.CAP_PROP_FPS else 35
            def read(self):
                self.reads += 1
                if self.reads <= 35:
                    return True, np.zeros((100, 100, 3), dtype=np.uint8)
                return False, None
            def release(self): pass

        def box(class_id, x, y, track_id):
            return SimpleNamespace(cls=[class_id], xyxy=np.array([[x - 5, y - 5, x + 5, y + 5]]),
                                   conf=[0.9], id=[track_id])

        frame_index = 0
        def detect(*args, **kwargs):
            nonlocal frame_index
            robot = SimpleNamespace(cls=[0], xyxy=np.array([[10, 10, 30, 50]]),
                                    conf=[0.95], id=[1])
            if frame_index < 16:
                balls = [box(1, 45, 45, 10)]
            else:
                # Same existing A, now listed after newly arrived B.
                balls = [box(1, 90, 45, 11), box(1, 45, 45, 10)]
            frame_index += 1
            return [SimpleNamespace(boxes=[robot, *balls])]

        model = SimpleNamespace(track=detect)
        logger = logging.getLogger("roi-reorder-integration-test")
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "robot.onnx"
            model_path.touch()
            with patch("control_vision.pipeline.tracking.cv2.VideoCapture", return_value=Capture()), \
                 patch("ultralytics.YOLO", return_value=model):
                _, _, events = track_video(
                    Path("match.mp4"), model_path,
                    [[0, 0], [100, 0], [100, 100], [0, 100]], [(1, 20, 50)],
                    30.0, lambda: False, lambda *_: None, logger,
                    goal_roi=[[0, 0], [100, 0], [100, 100], [0, 100]],
                )

        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0]["t_s"], 0.5)
        self.assertGreaterEqual(events[1]["t_s"], 1.0)

    def test_moving_roi_element_does_not_create_second_candidate(self):
        import numpy as np
        from types import SimpleNamespace

        class Capture:
            reads = 0
            def isOpened(self): return True
            def get(self, prop):
                import cv2
                return 30 if prop == cv2.CAP_PROP_FPS else 35
            def read(self):
                self.reads += 1
                return (True, np.zeros((100, 100, 3), dtype=np.uint8)) if self.reads <= 35 else (False, None)
            def release(self): pass

        for track_id in (10, None):
            with self.subTest(track_id=track_id):
                frame_index = 0
                def detect(*args, **kwargs):
                    nonlocal frame_index
                    robot = SimpleNamespace(cls=[0], xyxy=np.array([[10, 10, 30, 50]]),
                                            conf=[0.95], id=[1])
                    x = 45 if frame_index < 16 else 90
                    ball = SimpleNamespace(cls=[1], xyxy=np.array([[x - 5, 40, x + 5, 50]]),
                                           conf=[0.9], id=[track_id] if track_id is not None else None)
                    frame_index += 1
                    return [SimpleNamespace(boxes=[robot, ball])]

                with tempfile.TemporaryDirectory() as folder:
                    model_path = Path(folder) / "robot.onnx"
                    model_path.touch()
                    with patch("control_vision.pipeline.tracking.cv2.VideoCapture", return_value=Capture()), \
                         patch("ultralytics.YOLO", return_value=SimpleNamespace(track=detect)):
                        _, _, events = track_video(
                            Path("match.mp4"), model_path,
                            [[0, 0], [100, 0], [100, 100], [0, 100]], [(1, 20, 50)],
                            30.0, lambda: False, lambda *_: None,
                            logging.getLogger("roi-motion-test"),
                            goal_roi=[[0, 0], [100, 0], [100, 100], [0, 100]],
                        )
                self.assertEqual(len(events), 1)

    def test_assigns_one_to_one_by_minimum_total_distance(self):
        seeds = [(10, 100, 100), (20, 200, 100)]
        detections = [(1, 115, 100), (2, 180, 100)]
        matches = _match_initial_seeds(seeds, detections, set(), set(), 80)
        self.assertEqual({robot_id: item[0] for robot_id, item in matches.items()}, {10: 1, 20: 2})

    def test_rejects_detections_outside_seed_gate(self):
        seeds = [(10, 100, 100)]
        detections = [(1, 400, 100)]
        self.assertEqual(_match_initial_seeds(seeds, detections, set(), set(), 80), {})

    def test_never_reuses_an_assigned_detector_track(self):
        seeds = [(10, 100, 100), (20, 105, 100)]
        detections = [(1, 103, 100)]
        matches = _match_initial_seeds(seeds, detections, set(), {1}, 80)
        self.assertEqual(matches, {})


class ScoreConsensusTests(unittest.TestCase):
    def test_scoreboard_accepts_sqlite_rows_with_optional_alliance(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        try:
            event = connection.execute(
                "SELECT 'confirmed' AS status, NULL AS robot_id, 'red' AS scoring_alliance, "
                "2 AS points, 'hive_tip_teleop' AS event_type"
            ).fetchone()
            result = calculate_scoreboard([event], [])
            self.assertEqual(result["alliance_points"]["red"], 2)
        finally:
            connection.close()

    def test_prefers_consistent_final_score_over_one_ocr_outlier(self):
        result = _consensus([
            {"t_s": 1.0, "value": 90, "confidence": 0.99},
            {"t_s": 2.0, "value": 90, "confidence": 0.99},
            {"t_s": 3.0, "value": 80, "confidence": 0.99},
        ])
        self.assertEqual(result["value"], 90)
        self.assertAlmostEqual(result["agreement"], 2 / 3, places=3)

    def test_recent_final_score_wins_over_early_majority(self):
        rows = ([{"t_s": float(i), "value": 90, "confidence": .99} for i in range(8)] +
                [{"t_s": float(i), "value": 100, "confidence": .99} for i in range(8, 12)])
        self.assertEqual(_consensus(rows)["value"], 100)

    def test_ocr_does_not_lower_score_after_later_misreads(self):
        samples = [{"t_s": float(index), "value": value, "confidence": .9}
                   for index, value in enumerate((100, 90, 90))]
        result = _consensus(samples)
        self.assertEqual(result["value"], 100)
        self.assertTrue(result["later_lower_reading"])
        self.assertAlmostEqual(result["agreement"], 1 / 3, places=4)
        self.assertLess(result["confidence"], .4)
        later = _consensus(samples + [{"t_s": 3.0, "value": 90, "confidence": .9}])
        self.assertEqual(later["value"], 100)
        self.assertEqual(later["confidence"], 0)
        self.assertTrue(later["uncorroborated_prior_high"])

    def test_disqualification_does_not_resolve_flagged_evidence(self):
        result = calculate_scoreboard(
            [{"status": "flagged", "robot_id": None}] + [
                {"status": "confirmed", "robot_id": None, "event_alliance": "red",
                 "points": 2, "event_type": "hive_tip_teleop"} for _ in range(7)
            ], [], disqualified={"red"})
        self.assertFalse(result["match_resolved"])
        self.assertEqual(result["ranking_points"]["blue"]["WIN_TIE"], 0)
        self.assertEqual(result["ranking_points"]["red"]["total"], 0)

    def test_rejecting_every_candidate_resolves_zero_zero_tie(self):
        result = calculate_scoreboard([{"status": "rejected"}], [])
        self.assertTrue(result["match_resolved"])
        self.assertEqual(result["alliance_points"], {"red": 0, "blue": 0})
        self.assertEqual(result["ranking_points"]["red"]["WIN_TIE"], 1)
        self.assertEqual(result["ranking_points"]["blue"]["WIN_TIE"], 1)
        self.assertFalse(calculate_scoreboard([], [])["match_resolved"])

    def test_disqualified_robots_have_no_exported_video_points(self):
        robots = [{"id": 1, "team_number": 1234, "alliance": "red"},
                  {"id": 2, "team_number": 5678, "alliance": "blue"}]
        events = [{"status": "confirmed", "robot_id": robot_id, "points": 5,
                   "event_type": "hive_tip_teleop"} for robot_id in (1, 2)]
        result = calculate_scoreboard(events, robots, disqualified={"red"})
        self.assertEqual(result["alliance_points"], {"red": 0, "blue": 5})
        self.assertEqual(result["team_points"], {1: 0, 2: 5})

    def test_image_space_rois_are_not_field_zones(self):
        self.assertIsNone(zone_at(500, 650, {
            "scoreboard_red_roi": {"type": "scoreboard_red_roi", "points": [[450, 600], [600, 600], [600, 700], [450, 700]]}}))

    def test_polygon_roi_masks_pixels_outside_its_shape(self):
        import numpy as np

        class CaptureEngine:
            crop = None
            def __call__(self, image):
                self.crop = image.copy()
                return []

        frame = np.full((20, 20, 3), 255, dtype=np.uint8)
        engine = CaptureEngine()
        _read_digits(engine, frame, [[0, 0], [19, 0], [0, 19]])
        self.assertIsNotNone(engine.crop)
        # The lower-right triangle corner is outside the triangular ROI.
        self.assertEqual(int(engine.crop[-1, -1].max()), 0)


class DatabaseMigrationTests(unittest.TestCase):
    def test_unassigned_goal_candidate_survives_review_and_reprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            db = ProjectDatabase(Path(folder) / "project.sqlite")
            project_id = db.add_project("unassigned")
            video_id = db.add_video(project_id, Path("match.mp4"),
                                    {"duration_s": 120, "fps": 30, "width": 1280, "height": 720})
            robot_id = db.add_robot(video_id, 1234, "red", 10, 10)
            candidate = {"robot_id": None, "t_s": 12.0, "source": "goal_roi", "status": "flagged"}
            event_id = db.replace_analysis([robot_id], [], [candidate])[0]
            row = db.events_for_video(video_id)[0]
            self.assertIsNone(row["robot_id"])
            self.assertEqual(row["origin_robot_id"], 0)
            db.review_event(event_id, "confirmed", 2, robot_id, "flower_element", "medium", "red")
            self.assertEqual(db.replace_analysis([robot_id], [], [candidate]), [])
            self.assertEqual(len(db.events_for_video(video_id)), 1)
            db.close()

    def test_old_project_gets_default_yolo_mode_and_persists_new_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old-project.sqlite"
            old_schema = SCHEMA.replace(
                ", scoreboard_ocr_json TEXT, tracking_mode TEXT NOT NULL DEFAULT 'yolo', analysis_status TEXT NOT NULL DEFAULT 'ready'", "").replace(
                ", origin_robot_id INTEGER", "")
            connection = sqlite3.connect(path)
            connection.executescript(old_schema)
            connection.close()
            db = ProjectDatabase(path)
            project_id = db.add_project("migration")
            video_id = db.add_video(project_id, Path("match.mp4"),
                                    {"duration_s": 120, "fps": 30, "width": 1280, "height": 720})
            self.assertEqual(db.connection.execute(
                "SELECT tracking_mode FROM videos WHERE id=?", (video_id,)).fetchone()[0], "yolo")
            self.assertEqual(db.connection.execute(
                "SELECT analysis_status FROM videos WHERE id=?", (video_id,)).fetchone()[0], "ready")
            db.set_tracking_mode(video_id, "seeded_optical_flow")
            self.assertEqual(db.connection.execute(
                "SELECT tracking_mode FROM videos WHERE id=?", (video_id,)).fetchone()[0], "seeded_optical_flow")
            robot_id = db.add_robot(video_id, 1234, "red", 10, 10)
            manual_event = db.add_scoring_event(robot_id, 5.0)
            db.review_event(manual_event, "confirmed", 3, None, "park_auto", "high", "red")
            self.assertEqual(len(db.events_for_video(video_id)), 1)
            db.replace_analysis([robot_id], [], [])
            self.assertEqual(len(db.events_for_video(video_id)), 1)
            generated = {"robot_id": robot_id, "t_s": 8.0, "source": "goal_roi"}
            db.replace_analysis([robot_id], [], [generated])
            self.assertEqual(len(db.events_for_video(video_id)), 2)
            db.replace_analysis([robot_id], [], [generated])
            self.assertEqual(len(db.events_for_video(video_id)), 2)
            reviewed = {"robot_id": robot_id, "t_s": 12.0, "source": "zone_crossing"}
            generated_ids = db.replace_analysis([robot_id], [], [reviewed])
            db.review_event(generated_ids[0], "confirmed", 2, robot_id, "flower_element", "high", "red")
            db.replace_analysis([robot_id], [], [reviewed])
            matching = [row for row in db.events_for_video(video_id) if row["source"] == "zone_crossing"]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["status"], "confirmed")
            db.close()

    def test_failed_rerun_persists_invalid_status_and_blocks_export(self):
        import os
        from PySide6.QtWidgets import QApplication
        from control_vision.app.main_window import MainWindow

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "project.sqlite"
            db = ProjectDatabase(path)
            project_id = db.add_project("rerun")
            video_id = db.add_video(project_id, Path("match.mp4"),
                                    {"duration_s": 120, "fps": 30, "width": 1280, "height": 720})
            robot_id = db.add_robot(video_id, 1234, "red", 10, 10)
            db.replace_tracks([(robot_id, 0, 0.0, 2.0, 2.0, .9)])
            db.set_analysis_status(video_id, "running")
            window = MainWindow()
            window.db = db
            window.video_id = video_id
            window.tabs.setCurrentIndex(2)
            window.stats_table.setRowCount(1)
            window.tracks = {robot_id: [(0.0, 2.0, 2.0)]}
            window.processing_failed("model failed")
            self.assertEqual(window.tabs.currentIndex(), 1)
            self.assertFalse(window.tabs.isTabEnabled(2))
            self.assertFalse(window.tabs.isTabEnabled(6))
            self.assertEqual(window.stats_table.rowCount(), 0)
            self.assertEqual(window.tracks, {})
            with patch("control_vision.app.main_window.QMessageBox.warning") as warning:
                self.assertFalse(window._results_current())
                warning.assert_called_once()
            window.close()
            reopened = ProjectDatabase(path)
            self.assertEqual(reopened.connection.execute(
                "SELECT analysis_status FROM videos WHERE id=?", (video_id,)).fetchone()[0], "failed")
            self.assertEqual(len(reopened.tracks_for_video(video_id)), 1)
            reopened.close()

    def test_reprocessing_keeps_reassigned_event_without_collapsing_another_robot(self):
        with tempfile.TemporaryDirectory() as folder:
            db = ProjectDatabase(Path(folder) / "project.sqlite")
            project_id = db.add_project("reassignment")
            video_id = db.add_video(project_id, Path("match.mp4"),
                                    {"duration_s": 120, "fps": 30, "width": 1280, "height": 720})
            red_id = db.add_robot(video_id, 1234, "red", 10, 10)
            blue_id = db.add_robot(video_id, 5678, "blue", 20, 20)
            red_candidate = {"robot_id": red_id, "t_s": 12.0, "source": "zone_crossing"}
            blue_candidate = {"robot_id": blue_id, "t_s": 12.0, "source": "zone_crossing"}
            event_id = db.replace_analysis([red_id, blue_id], [], [red_candidate])[0]
            db.review_event(event_id, "confirmed", 2, blue_id, "flower_element", "high", "blue")
            inserted = db.replace_analysis([red_id, blue_id], [], [red_candidate, blue_candidate])
            self.assertEqual(len(inserted), 1)
            rows = [row for row in db.events_for_video(video_id) if row["source"] == "zone_crossing"]
            self.assertEqual(len(rows), 2)
            reviewed = next(row for row in rows if row["id"] == event_id)
            self.assertEqual((reviewed["robot_id"], reviewed["origin_robot_id"], reviewed["status"]),
                             (blue_id, red_id, "confirmed"))
            db.close()


class ReviewSelectionTests(unittest.TestCase):
    def test_unassigned_event_resets_previous_robot_selection(self):
        import os
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QTableWidgetItem
        from control_vision.app.main_window import MainWindow

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            window.review_robot.addItem("Unassigned", None)
            window.review_robot.addItem("1234 (red)", 1)
            window.review_robot.setCurrentIndex(1)
            window.events = [{"id": 7, "robot_id": None, "event_type": "unclassified",
                              "points": 0, "confidence": "low", "scoring_alliance": None}]
            window.review_table.setRowCount(1)
            item = QTableWidgetItem("7")
            item.setData(Qt.ItemDataRole.UserRole, 7)
            window.review_table.setItem(0, 0, item)
            window.load_review_selection(0, 0, -1, -1)
            self.assertEqual(window.review_robot.currentIndex(), 0)
            self.assertIsNone(window.review_robot.currentData())
        finally:
            window.close()


class ShutdownTests(unittest.TestCase):
    def test_active_analysis_blocks_project_and_video_switches(self):
        from types import SimpleNamespace
        from control_vision.app.main_window import MainWindow

        class ActiveThread:
            def isRunning(self): return True

        for slot in ("_thread", "_ocr_thread"):
            window = SimpleNamespace(_thread=None, _ocr_thread=None)
            setattr(window, slot, ActiveThread())
            with self.subTest(slot=slot), \
                 patch("control_vision.app.main_window.QMessageBox.information") as notice, \
                 patch("control_vision.app.main_window.QFileDialog.getExistingDirectory") as folder, \
                 patch("control_vision.app.main_window.QFileDialog.getOpenFileName") as file:
                MainWindow.new_project(window)
                MainWindow.open_project(window)
                MainWindow.import_video(window)
                MainWindow.load_saved_video(window)
                MainWindow.load_saved_video_id(window, 1)
                self.assertEqual(notice.call_count, 5)
                folder.assert_not_called()
                file.assert_not_called()

        # A queued completion callback can outlive QThread.isRunning().
        window = SimpleNamespace(_thread=None, _ocr_thread=None, _processing_active=True)
        with patch("control_vision.app.main_window.QMessageBox.information") as notice, \
             patch("control_vision.app.main_window.QFileDialog.getExistingDirectory") as folder:
            MainWindow.new_project(window)
            notice.assert_called_once()
            folder.assert_not_called()

    def test_close_waits_for_processing_before_database_close(self):
        from types import SimpleNamespace
        from control_vision.app.main_window import MainWindow

        calls = []
        class Signal:
            def disconnect(self, slot):
                calls.append("disconnect")
        class Worker:
            done = Signal()
            failed = Signal()
            def cancel(self):
                calls.append("cancel")
        class Thread:
            def isRunning(self): return True
            def quit(self): calls.append("quit")
            def wait(self, timeout):
                calls.append("wait")
                return True
        fake = SimpleNamespace(
            _worker=Worker(), _thread=Thread(), _ocr_worker=None, _ocr_thread=None,
            processing_done=lambda *_: None, processing_failed=lambda *_: None,
            scoreboard_ocr_done=lambda *_: None, scoreboard_ocr_failed=lambda *_: None,
            capture=SimpleNamespace(release=lambda: calls.append("release")),
            db=SimpleNamespace(close=lambda: calls.append("db_close")),
        )
        event = SimpleNamespace(accept=lambda: calls.append("accept"), ignore=lambda: calls.append("ignore"))
        MainWindow.closeEvent(fake, event)
        self.assertLess(calls.index("wait"), calls.index("db_close"))
        self.assertEqual(calls[-1], "accept")


class VideoIngestTests(unittest.TestCase):
    def test_unreadable_first_frame_is_not_saved_in_project(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from control_vision.app.main_window import MainWindow

        database = SimpleNamespace(add_video=Mock())
        window = SimpleNamespace(db=database, project_id=1)
        capture = SimpleNamespace(read=lambda: (False, None), release=lambda: None)
        with patch("control_vision.app.main_window.QFileDialog.getOpenFileName", return_value=("bad.mp4", "")), \
             patch("control_vision.app.main_window.QInputDialog.getText", side_effect=[("EV", True), ("Q3", True)]), \
             patch("control_vision.app.main_window.probe_video", return_value={"width": 1280, "height": 720, "fps": 30}), \
             patch("control_vision.app.main_window.cv2.VideoCapture", return_value=capture), \
             patch("control_vision.app.main_window.QMessageBox.critical"):
            MainWindow.import_video(window)
        database.add_video.assert_not_called()

    def test_nonnumeric_stream_duration_uses_readable_video_frame_count(self):
        class Capture:
            released = False
            def get(self, prop):
                import cv2
                return {cv2.CAP_PROP_FRAME_COUNT: 300, cv2.CAP_PROP_FPS: 30}[prop]
            def release(self):
                self.released = True

        capture = Capture()
        result = type("Result", (), {"returncode": 0, "stdout": (
            '{"streams":[{"avg_frame_rate":"30/1","duration":"N/A",'
            '"width":1280,"height":720}]}'
        )})()
        with patch("control_vision.pipeline.ingest.shutil.which", return_value="ffprobe"), \
             patch("control_vision.pipeline.ingest.subprocess.run", return_value=result), \
             patch("control_vision.pipeline.ingest.cv2.VideoCapture", return_value=capture):
            metadata = probe_video(Path("playable.mp4"))
        self.assertEqual(metadata["duration_s"], 10)
        self.assertTrue(capture.released)


class PreviewPhotoTests(unittest.TestCase):
    def test_default_tracking_patch_yields_exportable_photo_geometry(self):
        x0, y0, x1, y1 = _photo_crop_bounds((720, 1280, 3), 600, 510, 112, 88)
        self.assertGreaterEqual(x1 - x0, 200)
        self.assertGreaterEqual(y1 - y0, 200)

    def test_only_full_size_crops_are_written(self):
        import logging
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            photos = _save_best_photos({
                1: (1, 1.0, 0.0, np.zeros((199, 250, 3), dtype=np.uint8)),
                2: (1, 1.0, 0.0, np.zeros((200, 200, 3), dtype=np.uint8)),
            }, Path(folder), logging.getLogger("preview-photo-test"))
            self.assertEqual([row[0] for row in photos], [2])
            self.assertFalse((Path(folder) / "robot-1.png").exists())
            self.assertTrue((Path(folder) / "robot-2.png").is_file())


if __name__ == "__main__":
    unittest.main()
