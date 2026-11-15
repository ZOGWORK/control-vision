import gc
import os
import tempfile
import unittest
import sqlite3
from pathlib import Path
from unittest import mock

from control_vision.storage import database as database_module
from control_vision.storage.database import ProjectDatabase, SCHEMA

SPEC_ERA_SCHEMA = """
CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT);
CREATE TABLE videos (id INTEGER PRIMARY KEY, project_id INTEGER, path TEXT NOT NULL, duration_s REAL, fps REAL, width INTEGER, height INTEGER, event_code TEXT, match_number TEXT);
CREATE TABLE robots (id INTEGER PRIMARY KEY, video_id INTEGER, team_number INTEGER, alliance TEXT, color_hint TEXT);
CREATE TABLE tracks (id INTEGER PRIMARY KEY, robot_id INTEGER, frame INTEGER, t_s REAL, x_ft REAL, y_ft REAL, conf REAL);
CREATE TABLE cycles (id INTEGER PRIMARY KEY, robot_id INTEGER, start_t_s REAL, end_t_s REAL, pickup_zone TEXT, scoring_zone TEXT);
CREATE TABLE scoring_events (id INTEGER PRIMARY KEY, robot_id INTEGER, t_s REAL, points INTEGER, confidence TEXT, status TEXT, source TEXT);
CREATE TABLE clips (id INTEGER PRIMARY KEY, scoring_event_id INTEGER, path TEXT, start_t_s REAL, end_t_s REAL);
INSERT INTO projects VALUES (1,'legacy',NULL);
INSERT INTO videos(id,project_id,path) VALUES (1,1,'events.mp4'),(2,1,'tracks.mp4'),(3,1,'cycles.mp4'),(4,1,'empty.mp4');
INSERT INTO robots VALUES (1,1,1001,'red',NULL),(2,2,2002,'blue',NULL),(3,3,3003,'red',NULL),(4,4,4004,'blue',NULL);
INSERT INTO scoring_events VALUES (1,1,12.0,3,'high','confirmed','zone_crossing');
INSERT INTO clips VALUES (1,1,'clip.mp4',9.0,15.0);
INSERT INTO tracks VALUES (1,2,0,0.0,1.0,1.0,0.9);
INSERT INTO cycles VALUES (1,3,1.0,5.0,'loading_zone','garden');
"""


def _write_spec_era_project(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(SPEC_ERA_SCHEMA)
    connection.commit()
    connection.close()


class ProjectDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "project.sqlite"
        self.db = ProjectDatabase(self.path)
        self.addCleanup(self.db.close)
        project_id = self.db.add_project("practice")
        metadata = {"duration_s": 150, "fps": 30, "width": 1280, "height": 720}
        self.red_video = self.db.add_video(project_id, Path("red.mp4"), metadata)
        self.blue_video = self.db.add_video(project_id, Path("blue.mp4"), metadata)
        self.red_robot = self.db.add_robot(self.red_video, 1001, "red", 10, 10)
        self.blue_robot = self.db.add_robot(self.blue_video, 2002, "blue", 20, 20)

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

    def test_partial_review_edit_keeps_foul_recipient_and_classification(self):
        event_id = self.db.add_scoring_event(self.red_robot, 30.0)
        self.db.review_event(event_id, "confirmed", 10, event_type="foul_minor",
                             confidence="high", scoring_alliance="blue")
        self.db.review_event(event_id, "confirmed", 5)
        event = self.db.events_for_video(self.red_video)[0]
        self.assertEqual((event["points"], event["event_type"], event["confidence"],
                          event["scoring_alliance"], event["event_alliance"]),
                         (5, "foul_minor", "high", "blue", "blue"))
        self.db.review_event(event_id, "confirmed", 5, scoring_alliance=None)
        self.assertEqual(self.db.events_for_video(self.red_video)[0]["event_alliance"], "red")

    def test_review_rejects_unknown_status_confidence_and_alliance(self):
        event_id = self.db.add_scoring_event(self.red_robot, 30.0)
        for kwargs in ({"status": "approved"}, {"status": "confirmed", "confidence": "certain"},
                       {"status": "confirmed", "scoring_alliance": "green"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.db.review_event(event_id, points=1, **kwargs)
        self.assertEqual(self.db.events_for_video(self.red_video)[0]["status"], "flagged")

    def test_manual_event_requires_existing_robot(self):
        with self.assertRaises(ValueError):
            self.db.add_scoring_event(9999, 1.0)

    def test_generated_cycles_and_events_stay_with_target_video(self):
        cycle = {"robot_id": self.blue_robot, "start_t_s": 1, "end_t_s": 2,
                 "pickup_zone": "loading_zone", "scoring_zone": "garden"}
        with self.assertRaises(ValueError):
            self.db.replace_analysis([self.red_robot], [cycle], [])
        with self.assertRaises(ValueError):
            self.db.replace_analysis([self.red_robot], [], [{"robot_id": self.blue_robot, "t_s": 4}])
        self.assertEqual(self.db.events_for_video(self.red_video), [])

    def test_rerun_rejects_non_generated_sources(self):
        with self.assertRaises(ValueError):
            self.db.replace_analysis([self.red_robot], [], [{"robot_id": self.red_robot, "t_s": 50.0,
                                                             "source": "scoreboard_ocr"}])
        self.assertEqual(self.db.events_for_video(self.red_video), [])

    def test_rerun_keeps_clipped_flagged_event_and_drops_stale_unassigned_candidate(self):
        clipped, unassigned = self.db.replace_analysis(
            [self.red_robot], [], [{"robot_id": self.red_robot, "t_s": 20.0, "source": "goal_roi"},
                                   {"robot_id": None, "t_s": 40.0, "source": "goal_roi"}])
        self.assertNotEqual(clipped, unassigned)
        self.db.add_clip(clipped, Path("clip.mp4"), 17.0, 23.0)
        self.db.replace_analysis([self.red_robot], [], [])
        remaining = [event["id"] for event in self.db.events_for_video(self.red_video)]
        self.assertEqual(remaining, [clipped])
        self.assertEqual(len(self.db.clips_for_video(self.red_video)), 1)

    def test_reviewed_event_survives_rerun(self):
        candidate = {"robot_id": self.red_robot, "t_s": 12.0, "source": "zone_crossing"}
        event_id = self.db.replace_analysis([self.red_robot], [], [candidate])[0]
        self.db.review_event(event_id, "confirmed", 2, event_type="flower_owned_element")
        self.db.replace_analysis([self.red_robot], [], [candidate])
        events = self.db.events_for_video(self.red_video)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["id"], events[0]["status"], events[0]["points"]),
                         (event_id, "confirmed", 2))

    def test_close_candidates_remain_distinct_and_review_stays_with_nearest(self):
        first = {"robot_id": self.red_robot, "t_s": 12.0, "source": "goal_roi"}
        second = {"robot_id": self.red_robot, "t_s": 12.3, "source": "goal_roi"}
        ids = self.db.replace_analysis([self.red_robot], [], [first, second])
        self.assertEqual(len(ids), 2)
        self.db.review_event(ids[1], "confirmed", 2, event_type="flower_owned_element")
        self.db.replace_analysis([self.red_robot], [], [first, second])
        events = self.db.events_for_video(self.red_video)
        self.assertEqual(len(events), 2)
        reviewed = next(event for event in events if event["id"] == ids[1])
        self.assertEqual((reviewed["t_s"], reviewed["status"], reviewed["points"]),
                         (12.3, "confirmed", 2))
        self.assertEqual(next(event for event in events if event["t_s"] == 12.0)["status"], "flagged")

    def test_time_matching_preserves_maximum_reviewed_candidates(self):
        original = [{"robot_id": self.red_robot, "t_s": time, "source": "goal_roi"}
                    for time in (1.0, 1.5)]
        ids = self.db.replace_analysis([self.red_robot], [], original)
        for event_id in ids:
            self.db.review_event(event_id, "confirmed", 2, event_type="flower_owned_element")
        shifted = [{"robot_id": self.red_robot, "t_s": time, "source": "goal_roi"}
                   for time in (1.2, 0.6)]
        self.assertEqual(self.db.replace_analysis([self.red_robot], [], shifted), [])
        events = self.db.events_for_video(self.red_video)
        self.assertEqual(len(events), 2)
        self.assertEqual({event["id"] for event in events}, set(ids))
        self.assertTrue(all(event["status"] == "confirmed" for event in events))

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

    def test_simultaneous_legacy_reviews_map_to_their_original_robots(self):
        other_robot = self.db.add_robot(self.red_video, 1002, "red", 30, 30)
        for robot_id in (self.red_robot, other_robot):
            self.db.connection.execute(
                "INSERT INTO scoring_events(robot_id,video_id,t_s,status,points,source,origin_robot_id) "
                "VALUES (?,?,12,'confirmed',2,'zone_crossing',NULL)",
                (robot_id, self.red_video),
            )
        self.db.connection.commit()
        candidates = [{"robot_id": robot_id, "t_s": 12.0, "source": "zone_crossing"}
                      for robot_id in (self.red_robot, other_robot)]
        self.assertEqual(self.db.replace_analysis([self.red_robot, other_robot], [], candidates), [])
        events = self.db.events_for_video(self.red_video)
        self.assertEqual(len(events), 2)
        self.assertEqual({event["origin_robot_id"] for event in events}, {self.red_robot, other_robot})
        self.assertTrue(all(event["status"] == "confirmed" for event in events))

    def test_legacy_review_matches_shifted_rerun_candidate(self):
        self.db.connection.execute(
            "INSERT INTO scoring_events(robot_id,video_id,t_s,status,points,source,origin_robot_id) "
            "VALUES (?,?,12.0,'confirmed',3,'zone_crossing',NULL)", (self.red_robot, self.red_video))
        self.db.connection.commit()
        candidate = {"robot_id": self.red_robot, "t_s": 12.1, "source": "zone_crossing"}
        self.assertEqual(self.db.replace_analysis([self.red_robot], [], [candidate]), [])
        events = self.db.events_for_video(self.red_video)
        self.assertEqual([(event["status"], event["points"], event["origin_robot_id"]) for event in events],
                         [("confirmed", 3, self.red_robot)])


class LegacyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "spec.sqlite"
        _write_spec_era_project(self.path)

    def _assert_spec_era_migrated(self, db):
        statuses = dict(db.connection.execute("SELECT id,analysis_status FROM videos").fetchall())
        self.assertEqual(statuses, {1: "ready", 2: "ready", 3: "ready", 4: "unprocessed"})
        events = db.events_for_video(1)
        self.assertEqual([(row["video_id"], row["status"], row["points"]) for row in events],
                         [(1, "confirmed", 3)])
        self.assertEqual(len(db.clips_for_video(1)), 1)

    def test_existing_video_and_reviews_survive_schema_upgrade(self):
        path = Path(self.folder.name) / "legacy.sqlite"
        old_schema = SCHEMA.replace(
            ", scoreboard_ocr_json TEXT, tracking_mode TEXT NOT NULL DEFAULT 'yolo', analysis_status TEXT NOT NULL DEFAULT 'unprocessed'", ""
        ).replace(", origin_robot_id INTEGER", "")
        connection = sqlite3.connect(path)
        connection.executescript(old_schema)
        connection.execute("INSERT INTO projects(id,name) VALUES (1,'legacy')")
        connection.execute("INSERT INTO videos(id,project_id,path) VALUES (1,1,'match.mp4')")
        connection.execute("INSERT INTO videos(id,project_id,path) VALUES (2,1,'unprocessed.mp4')")
        connection.execute("INSERT INTO robots(id,video_id,team_number,alliance) VALUES (1,1,1001,'red')")
        connection.execute("INSERT INTO scoring_events(robot_id,video_id,t_s,status,points) VALUES (1,1,12,'confirmed',3)")
        connection.commit()
        connection.close()
        db = ProjectDatabase(path)
        try:
            video = db.connection.execute("SELECT tracking_mode,analysis_status FROM videos WHERE id=1").fetchone()
            self.assertEqual(tuple(video), ("yolo", "ready"))
            self.assertEqual(db.connection.execute(
                "SELECT analysis_status FROM videos WHERE id=2").fetchone()[0], "unprocessed")
            self.assertEqual(db.events_for_video(1)[0]["points"], 3)
            self.assertEqual(db.events_for_video(1)[0]["status"], "confirmed")
            fresh = db.add_video(1, Path("new.mp4"), {"duration_s": 120, "fps": 30,
                                                       "width": 1280, "height": 720})
            self.assertEqual(db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (fresh,)).fetchone()[0],
                             "unprocessed")
        finally:
            db.close()

    def test_spec_era_project_backfills_event_video_and_evidence_status(self):
        db = ProjectDatabase(self.path)
        try:
            self._assert_spec_era_migrated(db)
            self.assertEqual(db.replace_analysis([1], [], [{"robot_id": 1, "t_s": 12.0}]), [])
            self.assertEqual(len(db.events_for_video(1)), 1)
        finally:
            db.close()

    def test_failed_upgrade_rolls_back_and_retry_completes(self):
        real_connect = sqlite3.connect

        class FailingConnection(sqlite3.Connection):
            def execute(self, sql, *args):
                if sql.startswith("UPDATE videos SET analysis_status='ready'"):
                    raise sqlite3.OperationalError("simulated disk I/O error")
                return super().execute(sql, *args)

        with mock.patch.object(database_module.sqlite3, "connect",
                               lambda path: real_connect(path, factory=FailingConnection)):
            with self.assertRaises(sqlite3.OperationalError):
                ProjectDatabase(self.path)
        gc.collect()
        connection = sqlite3.connect(self.path)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(scoring_events)")}
        connection.close()
        self.assertNotIn("video_id", columns)
        db = ProjectDatabase(self.path)
        try:
            self._assert_spec_era_migrated(db)
        finally:
            db.close()

    def test_failed_open_releases_project_file(self):
        bad = Path(self.folder.name) / "bad.sqlite"
        connection = sqlite3.connect(bad)
        connection.executescript("CREATE TABLE videos (id INTEGER PRIMARY KEY, path TEXT);"
                                 "CREATE VIEW robots AS SELECT 1 AS id;")
        connection.close()
        with self.assertRaises(sqlite3.OperationalError):
            ProjectDatabase(bad)
        os.remove(bad)
        self.assertFalse(bad.exists())


if __name__ == "__main__":
    unittest.main()
