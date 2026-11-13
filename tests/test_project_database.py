import tempfile
import unittest
from pathlib import Path

from control_vision.storage.database import ProjectDatabase


class ProjectDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "project.sqlite"
        self.db = ProjectDatabase(self.path)
        project_id = self.db.add_project("practice")
        metadata = {"duration_s": 150, "fps": 30, "width": 1280, "height": 720}
        self.red_video = self.db.add_video(project_id, Path("red.mp4"), metadata)
        self.blue_video = self.db.add_video(project_id, Path("blue.mp4"), metadata)
        self.red_robot = self.db.add_robot(self.red_video, 1001, "red", 10, 10)
        self.blue_robot = self.db.add_robot(self.blue_video, 2002, "blue", 20, 20)

    def tearDown(self):
        self.db.close()
        self.folder.cleanup()

    def test_new_video_is_unprocessed_and_state_survives_reopen(self):
        row = self.db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (self.red_video,)).fetchone()
        self.assertEqual(row[0], "unprocessed")
        self.db.set_analysis_status(self.red_video, "ready")
        self.db.set_official_scores(self.red_video, 180, 103)
        reopened = ProjectDatabase(self.path)
        try:
            row = reopened.connection.execute("SELECT analysis_status,official_red,official_blue FROM videos WHERE id=?", (self.red_video,)).fetchone()
            self.assertEqual(tuple(row), ("ready", 180, 103))
        finally:
            reopened.close()

    def test_review_preserves_robot_and_rejects_cross_video_assignment(self):
        event_id = self.db.add_scoring_event(self.red_robot, 12.0)
        self.db.review_event(event_id, "confirmed", 3, event_type="leave_auto")
        event = self.db.events_for_video(self.red_video)[0]
        self.assertEqual((event["robot_id"], event["status"], event["points"]), (self.red_robot, "confirmed", 3))
        with self.assertRaises(ValueError):
            self.db.review_event(event_id, "confirmed", 3, self.blue_robot)
        self.assertEqual(self.db.events_for_video(self.blue_video), [])
        self.db.review_event(event_id, "confirmed", 3, None)
        self.assertIsNone(self.db.events_for_video(self.red_video)[0]["robot_id"])

    def test_generated_cycles_and_events_stay_with_target_video(self):
        cycle = {"robot_id": self.blue_robot, "start_t_s": 1, "end_t_s": 2,
                 "pickup_zone": "loading_zone", "scoring_zone": "garden"}
        with self.assertRaises(ValueError):
            self.db.replace_analysis([self.red_robot], [cycle], [])
        with self.assertRaises(ValueError):
            self.db.replace_analysis([self.red_robot], [], [{"robot_id": self.blue_robot, "t_s": 4}])
        self.assertEqual(self.db.events_for_video(self.red_video), [])

    def test_reviewed_event_survives_rerun(self):
        candidate = {"robot_id": self.red_robot, "t_s": 12.0, "source": "zone_crossing"}
        event_id = self.db.replace_analysis([self.red_robot], [], [candidate])[0]
        self.db.review_event(event_id, "confirmed", 2, event_type="flower_owned_element")
        self.db.replace_analysis([self.red_robot], [], [candidate])
        events = self.db.events_for_video(self.red_video)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["id"], events[0]["status"], events[0]["points"]),
                         (event_id, "confirmed", 2))

    def test_reassignment_keeps_independent_simultaneous_robot_candidate(self):
        other_robot = self.db.add_robot(self.red_video, 1002, "red", 30, 30)
        first = {"robot_id": self.red_robot, "t_s": 12.0, "source": "zone_crossing"}
        simultaneous = {"robot_id": other_robot, "t_s": 12.0, "source": "zone_crossing"}
        reviewed_id = self.db.replace_analysis([self.red_robot, other_robot], [], [first])[0]
        self.db.review_event(reviewed_id, "confirmed", 2, other_robot, "flower_owned_element")
        self.db.replace_analysis([self.red_robot, other_robot], [], [first, simultaneous])
        events = self.db.events_for_video(self.red_video)
        self.assertEqual(len(events), 2)
        self.assertEqual({event["origin_robot_id"] for event in events}, {self.red_robot, other_robot})
        self.assertEqual(next(event for event in events if event["id"] == reviewed_id)["status"], "confirmed")


if __name__ == "__main__":
    unittest.main()
