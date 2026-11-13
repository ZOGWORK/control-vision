import json
import tempfile
import unittest
from pathlib import Path

from control_vision.game.scoreboard import calculate_scoreboard
from control_vision.scout.exports import export_scout_json
from control_vision.storage.database import ProjectDatabase


class CoreScoringTests(unittest.TestCase):
    def setUp(self):
        self.robots = [{"id": 1, "team_number": 1001, "alliance": "red"},
                       {"id": 2, "team_number": 2002, "alliance": "blue"}]

    def test_pending_review_cannot_claim_a_winner(self):
        events = [{"robot_id": 1, "status": "confirmed", "points": 5, "event_type": "nectar_floor"},
                  {"robot_id": 2, "status": "flagged", "points": 10, "event_type": "nectar_floor"}]
        board = calculate_scoreboard(events, self.robots)
        self.assertFalse(board["match_resolved"])
        self.assertEqual(board["alliance_points"], {"red": 5, "blue": 0})
        self.assertEqual(board["ranking_points"]["red"]["WIN_TIE"], 0)

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


if __name__ == "__main__":
    unittest.main()
