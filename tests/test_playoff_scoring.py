import unittest

from control_vision.game.scoreboard import calculate_playoff_scoreboard


class PlayoffScoringTests(unittest.TestCase):
    def setUp(self):
        self.robots = [{"id": 1, "alliance": "red"}, {"id": 2, "alliance": "blue"}]
        self.events = [{"robot_id": 1, "status": "confirmed", "points": 20,
                        "event_type": "park_teleop"}]

    def test_playoff_has_no_ranking_points_and_dq_overrides_score(self):
        board = calculate_playoff_scoreboard(self.events, self.robots,
                                             disqualified={"red"})
        self.assertEqual(board["alliance_points"], {"red": 0, "blue": 0})
        self.assertEqual(board["playoff_outcome"],
                         {"resolved": True, "winner": "blue", "tie": False})
        for alliance in ("red", "blue"):
            self.assertEqual(board["ranking_points"][alliance]["total"], 0)
            self.assertEqual(board["ranking_points"][alliance]["WIN_TIE"], 0)
        self.assertEqual(board["team_ranking_points"], {1: 0, 2: 0})

    def test_double_dq_requires_chronology_or_simultaneous_ruling(self):
        unknown = calculate_playoff_scoreboard(self.events, self.robots,
                                               disqualified={"red", "blue"})
        self.assertEqual(unknown["playoff_outcome"],
                         {"resolved": False, "winner": None, "tie": False})
        for first, winner in (("red", "blue"), ("blue", "red")):
            with self.subTest(first=first):
                board = calculate_playoff_scoreboard(self.events, self.robots,
                                                     disqualified={"red", "blue"},
                                                     first_disqualified=first)
                self.assertEqual(board["playoff_outcome"],
                                 {"resolved": True, "winner": winner, "tie": False})
        simultaneous = calculate_playoff_scoreboard(self.events, self.robots,
                                                    disqualified={"red", "blue"},
                                                    simultaneous_disqualification=True)
        self.assertEqual(simultaneous["playoff_outcome"],
                         {"resolved": True, "winner": None, "tie": True})

    def test_reviewed_zero_zero_match_can_tie_without_event_rows(self):
        board = calculate_playoff_scoreboard([], self.robots, match_final=True)
        self.assertEqual(board["playoff_outcome"],
                         {"resolved": True, "winner": None, "tie": True})
        pending = calculate_playoff_scoreboard([{"robot_id": 1, "status": "flagged",
                                                 "points": 5}], self.robots,
                                               match_final=True)
        self.assertFalse(pending["playoff_outcome"]["resolved"])
        unfinalized = calculate_playoff_scoreboard(self.events, self.robots)
        self.assertFalse(unfinalized["playoff_outcome"]["resolved"])

    def test_invalid_referee_ruling_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown playoff alliance"):
            calculate_playoff_scoreboard([], self.robots, disqualified={"green"})
        with self.assertRaisesRegex(ValueError, "First disqualified"):
            calculate_playoff_scoreboard([], self.robots, disqualified={"red"},
                                         first_disqualified="blue")
        with self.assertRaisesRegex(ValueError, "First disqualified"):
            calculate_playoff_scoreboard([], self.robots, disqualified={"red"},
                                         first_disqualified="red")
        with self.assertRaisesRegex(ValueError, "Simultaneous DQ"):
            calculate_playoff_scoreboard([], self.robots, disqualified={"red"},
                                         simultaneous_disqualification=True)


if __name__ == "__main__":
    unittest.main()
