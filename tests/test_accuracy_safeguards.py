import unittest

from control_vision.pipeline.scoreboard_ocr import _consensus
from control_vision.pipeline.tracking import _match_initial_seeds


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


if __name__ == "__main__":
    unittest.main()
