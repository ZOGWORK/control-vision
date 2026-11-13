import json
import unittest
from pathlib import Path

from control_vision.game.scoreboard import calculate_scoreboard, classify_match_time


CONFIG = Path(__file__).resolve().parents[1] / "control_vision" / "config"


class BiobuzzRulesTests(unittest.TestCase):
    def setUp(self):
        self.robots = [{"id": 1, "alliance": "red"}, {"id": 2, "alliance": "blue"}]

    def test_rule_config_has_robot_class_and_match_periods(self):
        classes = json.loads((CONFIG / "game_classes_biobuzz.json").read_text(encoding="utf-8"))
        scoring = json.loads((CONFIG / "scoring_biobuzz.json").read_text(encoding="utf-8"))
        self.assertEqual(classes["classes"][0], {"id": 0, "name": "robot"})
        self.assertEqual({key: scoring["match_periods_s"][key] for key in ("auto", "transition", "teleop")},
                         {"auto": 30, "transition": 8, "teleop": 120})

    def test_review_status_gates_winner_and_confirmed_points(self):
        for unresolved in ("flagged", "pending", "unknown"):
            with self.subTest(unresolved=unresolved):
                events = [{"robot_id": 1, "status": "confirmed", "points": 5},
                          {"robot_id": 2, "status": unresolved, "points": 10}]
                result = calculate_scoreboard(events, self.robots)
                self.assertEqual(result["alliance_points"], {"red": 5, "blue": 0})
                self.assertFalse(result["match_resolved"])
                self.assertEqual(result["ranking_points"]["red"]["WIN_TIE"], 0)

    def test_empty_final_queue_ties_and_disqualification_clears_points(self):
        tie = calculate_scoreboard([], self.robots)
        self.assertEqual(tie["ranking_points"]["red"]["WIN_TIE"], 1)
        self.assertEqual(tie["ranking_points"]["blue"]["WIN_TIE"], 1)
        self.assertEqual(tie["team_points"], {1: 0, 2: 0})
        event = [{"robot_id": 1, "status": "confirmed", "points": 20, "event_type": "park_teleop"}]
        dq = calculate_scoreboard(event, self.robots, disqualified={"red"})
        self.assertEqual(dq["alliance_points"], {"red": 0, "blue": 0})
        self.assertEqual(dq["team_points"][1], 0)
        self.assertEqual(dq["ranking_points"]["red"]["total"], 0)
        self.assertEqual(dq["ranking_points"]["blue"]["WIN_TIE"], 3)

    def test_reviewed_alliance_overrides_stale_inferred_alliance(self):
        event = [{"robot_id": 1, "status": "confirmed", "points": 5,
                  "event_alliance": "red", "scoring_alliance": "blue"}]
        result = calculate_scoreboard(event, self.robots)
        self.assertEqual(result["alliance_points"], {"red": 0, "blue": 5})
        self.assertEqual(result["ranking_points"]["blue"]["WIN_TIE"], 3)
        self.assertEqual(result["team_points"][1], 5)

    def test_one_shot_event_and_robot_iterators_score_once(self):
        events = iter([{"robot_id": 1, "status": "confirmed", "points": 5}])
        result = calculate_scoreboard(events, iter(self.robots))
        self.assertEqual(result["alliance_points"], {"red": 5, "blue": 0})
        self.assertEqual(result["team_points"][1], 5)
        self.assertEqual(result["team_ranking_points"][1], result["ranking_points"]["red"]["total"])

    def test_period_boundaries(self):
        periods = {"auto": 30, "transition": 8, "teleop": 120}
        self.assertEqual(classify_match_time(0, periods), ("AUTO", 30))
        self.assertEqual(classify_match_time(30, periods), ("TRANSITION", 8))
        self.assertEqual(classify_match_time(38, periods), ("TELEOP", 120))
        self.assertEqual(classify_match_time(158, periods), ("POST-MATCH", 0))


if __name__ == "__main__":
    unittest.main()
