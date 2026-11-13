import unittest
import sqlite3
import tempfile
from pathlib import Path

from control_vision.pipeline.scoreboard_ocr import _consensus, _read_digits
from control_vision.pipeline.tracking import _match_initial_seeds
from control_vision.storage.database import ProjectDatabase, SCHEMA


class SeedMatchingTests(unittest.TestCase):
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
    def test_prefers_consistent_final_score_over_one_ocr_outlier(self):
        result = _consensus([
            {"t_s": 1.0, "value": 90, "confidence": 0.99},
            {"t_s": 2.0, "value": 90, "confidence": 0.99},
            {"t_s": 3.0, "value": 80, "confidence": 0.99},
        ])
        self.assertEqual(result["value"], 90)
        self.assertAlmostEqual(result["agreement"], 2 / 3, places=3)

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
                ", scoreboard_ocr_json TEXT, tracking_mode TEXT NOT NULL DEFAULT 'yolo'", "")
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
            db.close()


if __name__ == "__main__":
    unittest.main()
