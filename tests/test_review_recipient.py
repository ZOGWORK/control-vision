"""Confirmed points must always credit an alliance (found in an end-to-end app run)."""
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from control_vision.scout.exports import match_scoreboard
from control_vision.storage.database import ProjectDatabase

META = {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720}


def _seed(db):
    project = db.add_project("event")
    video = db.add_video(project, Path("Q1.mp4"), META, "EV", "Q1")
    robots = {team: db.add_robot(video, team, alliance, 10, 10)
              for team, alliance in ((1001, "red"), (2001, "blue"))}
    return project, video, robots


def _unattributed_event(db, robot_id, t_s=50.0):
    """A flagged candidate with no attributed robot, like a goal-ROI candidate."""
    event_id = db.add_scoring_event(robot_id, t_s)
    db.review_event(event_id, "flagged", 0, None)
    return event_id


def _totals(db, video):
    row = dict(db.connection.execute("SELECT * FROM videos WHERE id=?", (video,)).fetchone())
    board = match_scoreboard(row, [dict(r) for r in db.robots_for_video(video)],
                             [dict(e) for e in db.events_for_video(video)])
    return board["alliance_points"]


class ReviewRecipientDatabaseTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = ProjectDatabase(Path(folder.name) / "project.sqlite")
        self.addCleanup(self.db.close)
        self.project, self.video, self.robots = _seed(self.db)

    def event(self, event_id):
        return next(e for e in self.db.events_for_video(self.video) if e["id"] == event_id)

    def test_confirming_points_with_no_robot_or_recipient_is_refused(self):
        event_id = _unattributed_event(self.db, self.robots[1001])
        with self.assertRaisesRegex(ValueError, "attributed robot or the alliance"):
            self.db.review_event(event_id, "confirmed", 20, None, "major_foul", "high", None)
        event = self.event(event_id)
        self.assertEqual((event["status"], event["points"], event["event_type"]), ("flagged", 0, "unclassified"))
        self.assertEqual(_totals(self.db, self.video), {"red": 0, "blue": 0})

    def test_a_recipient_alliance_alone_credits_the_points(self):
        event_id = _unattributed_event(self.db, self.robots[1001])
        self.db.review_event(event_id, "confirmed", 20, None, "major_foul", "high", "blue")
        self.assertEqual(_totals(self.db, self.video), {"red": 0, "blue": 20})

    def test_an_attributed_robot_alone_credits_its_alliance(self):
        event_id = _unattributed_event(self.db, self.robots[1001])
        self.db.review_event(event_id, "confirmed", 3, self.robots[1001], "leave_auto", "high", None)
        self.assertEqual(_totals(self.db, self.video), {"red": 3, "blue": 0})

    def test_unchanged_fields_count_as_the_stored_recipient(self):
        # A points-only edit keeps the stored robot, so it is still credited.
        event_id = self.db.add_scoring_event(self.robots[2001], 60.0)
        self.db.review_event(event_id, "confirmed", 5)
        self.assertEqual(_totals(self.db, self.video), {"red": 0, "blue": 5})

    def test_zero_points_and_rejections_need_no_recipient(self):
        zero = _unattributed_event(self.db, self.robots[1001], 10.0)
        rejected = _unattributed_event(self.db, self.robots[1001], 20.0)
        self.db.review_event(zero, "confirmed", 0, None, "unclassified", "low", None)
        self.db.review_event(rejected, "rejected", 20, None, "major_foul", "low", None)
        self.assertEqual((self.event(zero)["status"], self.event(rejected)["status"]), ("confirmed", "rejected"))


class ReviewRecipientUiTests(unittest.TestCase):
    warning: Any
    information: Any

    def setUp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from control_vision.app.main_window import MainWindow
        self.app = QApplication.instance() or QApplication([])
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = ProjectDatabase(Path(folder.name) / "project.sqlite")
        self.project, self.video, self.robots = _seed(self.db)
        self.event_id = _unattributed_event(self.db, self.robots[1001])
        self.db.set_analysis_status(self.video, "ready")
        # Modal dialogs would block an offscreen run; record them instead.
        for name in ("warning", "information"):
            mocked = patch(f"control_vision.app.main_window.QMessageBox.{name}")
            setattr(self, name, mocked.start())
            self.addCleanup(mocked.stop)
        self.window = MainWindow()
        self.window.db, self.window.project_id, self.window.video_id = self.db, self.project, self.video
        self.window.refresh_result_tables()

    def tearDown(self):
        self.window.db = None
        self.window.close()
        self.db.close()

    def select_event(self):
        window = self.window
        row = next(r for r in range(window.review_table.rowCount())
                   if window.review_table.item(r, 0).data(0x0100) == self.event_id)
        window.review_table.setCurrentCell(row, 0)
        window.review_type.setCurrentIndex(window.review_type.findData("major_foul"))
        window.review_points.setValue(20)

    def test_confirming_unattributed_points_warns_and_keeps_the_event_flagged(self):
        self.select_event()
        self.assertIsNone(self.window.review_robot.currentData())
        self.assertIsNone(self.window.review_alliance.currentData())
        self.window.review_selected("confirmed")
        self.warning.assert_called_once()
        self.assertIn("attributed robot or the alliance", self.warning.call_args[0][2])
        event = next(e for e in self.db.events_for_video(self.video) if e["id"] == self.event_id)
        self.assertEqual(event["status"], "flagged")

    def test_choosing_the_recipient_alliance_confirms_and_counts(self):
        self.select_event()
        self.window.review_alliance.setCurrentIndex(self.window.review_alliance.findData("blue"))
        self.window.review_selected("confirmed")
        self.warning.assert_not_called()
        self.assertEqual(_totals(self.db, self.video), {"red": 0, "blue": 20})

    def test_add_event_asks_for_a_robot_when_robots_are_seeded(self):
        window = self.window
        window.review_robot.setCurrentIndex(0)
        window.add_manual_event()
        self.information.assert_called_once()
        self.assertEqual(self.information.call_args[0][1], "Choose a robot")
        self.assertEqual(len(self.db.events_for_video(self.video)), 1)
        window.review_robot.setCurrentIndex(window.review_robot.findData(self.robots[2001]))
        window.add_manual_event()
        self.assertEqual(len(self.db.events_for_video(self.video)), 2)

    def test_add_event_without_seeded_robots_asks_to_seed(self):
        window = self.window
        window.review_robot.clear()
        window.review_robot.addItem("Unassigned", None)
        window.add_manual_event()
        self.assertEqual(self.information.call_args[0][1], "Set up robot identities")


if __name__ == "__main__":
    unittest.main()
