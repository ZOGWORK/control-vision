import unittest
import sqlite3
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch

from control_vision.pipeline.scoreboard_ocr import _consensus, _read_digits
from control_vision.pipeline.tracking import _match_initial_seeds, _new_roi_candidate
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
    def test_goal_element_reacquisition_does_not_repeat_candidate(self):
        recent = [(100.0, 100.0, 1.0)]
        self.assertFalse(_new_roi_candidate(105.0, 99.0, 2.0, recent))
        self.assertTrue(_new_roi_candidate(180.0, 100.0, 2.0, recent))
        self.assertTrue(_new_roi_candidate(100.0, 100.0, 4.0, recent))

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

    def test_disqualification_does_not_resolve_flagged_evidence(self):
        result = calculate_scoreboard(
            [{"status": "flagged", "robot_id": None}] + [
                {"status": "confirmed", "robot_id": None, "event_alliance": "red",
                 "points": 2, "event_type": "hive_tip_teleop"} for _ in range(7)
            ], [], disqualified={"red"})
        self.assertFalse(result["match_resolved"])
        self.assertEqual(result["ranking_points"]["blue"]["WIN_TIE"], 0)
        self.assertEqual(result["ranking_points"]["red"]["total"], 0)

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
    def test_old_project_gets_default_yolo_mode_and_persists_new_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old-project.sqlite"
            old_schema = SCHEMA.replace(
                ", scoreboard_ocr_json TEXT, tracking_mode TEXT NOT NULL DEFAULT 'yolo'", "").replace(
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


class ShutdownTests(unittest.TestCase):
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
