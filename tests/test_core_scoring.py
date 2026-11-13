import json
import tempfile
import unittest
from pathlib import Path

from control_vision.game.scoreboard import calculate_scoreboard
from control_vision.pipeline.cycles import count_zone_cycles
from control_vision.scout.exports import export_scout_json
from control_vision.storage.database import ProjectDatabase


class CoreScoringTests(unittest.TestCase):
    def setUp(self):
        self.robots = [{"id": 1, "team_number": 1001, "alliance": "red"},
                       {"id": 2, "team_number": 2002, "alliance": "blue"}]

    def test_pending_review_cannot_claim_a_winner(self):
        for status in ("flagged", "pending", "unknown"):
            with self.subTest(status=status):
                events = [{"robot_id": 1, "status": "confirmed", "points": 5, "event_type": "nectar_floor"},
                          {"robot_id": 2, "status": status, "points": 10, "event_type": "nectar_floor"}]
                board = calculate_scoreboard(events, self.robots)
                self.assertFalse(board["match_resolved"])
                self.assertEqual(board["alliance_points"], {"red": 5, "blue": 0})
                self.assertEqual(board["ranking_points"]["red"]["WIN_TIE"], 0)

    def test_empty_completed_review_queue_resolves_zero_zero_tie(self):
        board = calculate_scoreboard([], self.robots)
        self.assertTrue(board["match_resolved"])
        self.assertEqual(board["ranking_points"]["red"]["WIN_TIE"], 1)
        self.assertEqual(board["ranking_points"]["blue"]["WIN_TIE"], 1)

    def test_legacy_confirmed_event_without_type_still_scores(self):
        board = calculate_scoreboard([{"robot_id": 1, "status": "confirmed", "points": 5}], self.robots)
        self.assertEqual(board["alliance_points"]["red"], 5)

    def test_disqualification_clears_alliance_and_robot_points(self):
        events = [{"robot_id": 1, "status": "confirmed", "points": 20, "event_type": "park_teleop"},
                  {"robot_id": 2, "status": "confirmed", "points": 5, "event_type": "nectar_floor"}]
        board = calculate_scoreboard(events, self.robots, disqualified={"red"})
        self.assertEqual(board["alliance_points"], {"red": 0, "blue": 5})
        self.assertEqual(board["team_points"][1], 0)
        self.assertEqual(board["team_ranking_points"][1], 0)
        self.assertEqual(board["ranking_points"]["blue"]["WIN_TIE"], 3)

    def test_project_status_and_review_survive_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.db"
            db = ProjectDatabase(path)
            project = db.add_project("practice")
            video = db.add_video(project, Path("match.mp4"),
                                 {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720})
            self.assertEqual(db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (video,)).fetchone()[0], "unprocessed")
            robot = db.add_robot(video, 1001, "red", 50, 60)
            event = db.add_scoring_event(robot, 42.0)
            db.review_event(event, "confirmed", 5, robot, "nectar_floor", "high")
            db.set_analysis_status(video, "failed")
            db.close()
            reopened = ProjectDatabase(path)
            self.assertEqual(reopened.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (video,)).fetchone()[0], "failed")
            self.assertEqual(reopened.events_for_video(video)[0]["points"], 5)
            reopened.close()

    def test_event_cannot_be_reassigned_across_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            db = ProjectDatabase(Path(directory) / "event.db")
            project = db.add_project("event")
            metadata = {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720}
            first_video = db.add_video(project, Path("first.mp4"), metadata)
            second_video = db.add_video(project, Path("second.mp4"), metadata)
            first_robot = db.add_robot(first_video, 1001, "red", 50, 60)
            second_robot = db.add_robot(second_video, 2002, "blue", 70, 80)
            event = db.add_scoring_event(first_robot, 42.0)
            with self.assertRaisesRegex(ValueError, "another video"):
                db.review_event(event, "confirmed", 5, second_robot)
            self.assertEqual(len(db.events_for_video(first_video)), 1)
            self.assertEqual(len(db.events_for_video(second_video)), 0)
            db.close()

    def test_generated_events_cannot_cross_analysis_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            db = ProjectDatabase(Path(directory) / "event.db")
            project = db.add_project("event")
            metadata = {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720}
            first_video = db.add_video(project, Path("first.mp4"), metadata)
            second_video = db.add_video(project, Path("second.mp4"), metadata)
            first_robot = db.add_robot(first_video, 1001, "red", 50, 60)
            second_robot = db.add_robot(second_video, 2002, "blue", 70, 80)
            candidate = {"robot_id": second_robot, "t_s": 42.0, "source": "goal_roi"}
            with self.assertRaisesRegex(ValueError, "outside this analysis"):
                db.replace_analysis([first_robot], [], [candidate])
            self.assertEqual(db.events_for_video(first_video), [])
            self.assertEqual(db.events_for_video(second_video), [])
            db.close()

    def test_review_keeps_robot_unless_explicitly_unassigned(self):
        with tempfile.TemporaryDirectory() as directory:
            db = ProjectDatabase(Path(directory) / "event.db")
            project = db.add_project("event")
            video = db.add_video(project, Path("match.mp4"),
                                 {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720})
            robot = db.add_robot(video, 1001, "red", 50, 60)
            event = db.add_scoring_event(robot, 42.0)
            db.review_event(event, "confirmed", 5)
            self.assertEqual(db.events_for_video(video)[0]["robot_id"], robot)
            db.review_event(event, "flagged", 0, None)
            self.assertIsNone(db.events_for_video(video)[0]["robot_id"])
            db.close()

    def test_scout_export_uses_reviewed_disqualification_aware_points(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "scout.json"
            video = {"path": "match.mp4", "event_code": "TEST", "match_number": "Q1",
                     "red_disqualified": 1, "blue_disqualified": 0,
                     "official_red": 100, "official_blue": 90}
            events = [{"robot_id": 1, "status": "confirmed", "points": 9, "event_type": "nectar_floor"}]
            export_scout_json(target, video, self.robots, [], [], events, [], [], {})
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["alliance_results"]["red"]["points_video"], 0)
            self.assertEqual(payload["teams"][0]["points_video"], 0)
            self.assertEqual(payload["official_alliance_scores"], {"red": 100, "blue": 90})

    def test_biobuzz_zone_types_produce_reviewable_cycles(self):
        for scoring_type in ("garden", "flower_scoring_volume", "hive"):
            with self.subTest(scoring_type=scoring_type):
                zones = {
                    "load": {"type": "loading_zone", "points": [[0, 0], [2, 0], [2, 2], [0, 2]]},
                    "goal": {"type": scoring_type, "points": [[8, 8], [10, 8], [10, 10], [8, 10]]},
                }
                rows = [(1, 0, 0.0, 1.0, 1.0, 0.9), (1, 1, 1.0, 4.0, 4.0, 0.9),
                        (1, 2, 2.0, 9.0, 9.0, 0.9)]
                cycles, events = count_zone_cycles(rows, zones)
                self.assertEqual(len(cycles), 1)
                self.assertEqual(events[0]["status"], "flagged")
                self.assertEqual(events[0]["points"], 0)


if __name__ == "__main__":
    unittest.main()
