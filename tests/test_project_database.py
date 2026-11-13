import tempfile
import unittest
import sqlite3
from pathlib import Path

from control_vision.storage.database import ProjectDatabase, SCHEMA


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


class LegacyMigrationTests(unittest.TestCase):
    def test_existing_video_and_reviews_survive_schema_upgrade(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.sqlite"
            old_schema = SCHEMA.replace(
                ", scoreboard_ocr_json TEXT, tracking_mode TEXT NOT NULL DEFAULT 'yolo', analysis_status TEXT NOT NULL DEFAULT 'unprocessed'", ""
            ).replace(", origin_robot_id INTEGER", "")
            connection = sqlite3.connect(path)
            connection.executescript(old_schema)
            connection.execute("INSERT INTO projects(id,name) VALUES (1,'legacy')")
            connection.execute("INSERT INTO videos(id,project_id,path) VALUES (1,1,'match.mp4')")
            connection.execute("INSERT INTO robots(id,video_id,team_number,alliance) VALUES (1,1,1001,'red')")
            connection.execute("INSERT INTO scoring_events(robot_id,video_id,t_s,status,points) VALUES (1,1,12,'confirmed',3)")
            connection.commit()
            connection.close()
            db = ProjectDatabase(path)
            try:
                video = db.connection.execute("SELECT tracking_mode,analysis_status FROM videos WHERE id=1").fetchone()
                self.assertEqual(tuple(video), ("yolo", "ready"))
                self.assertEqual(db.events_for_video(1)[0]["points"], 3)
                self.assertEqual(db.events_for_video(1)[0]["status"], "confirmed")
                fresh = db.add_video(1, Path("new.mp4"), {"duration_s": 120, "fps": 30,
                                                           "width": 1280, "height": 720})
                self.assertEqual(db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (fresh,)).fetchone()[0],
                                 "unprocessed")
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
