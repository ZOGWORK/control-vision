import csv
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from control_vision.scout.exports import export_scout_json, export_stats_csv, match_scoreboard
from control_vision.storage.database import ProjectDatabase

META = {"duration_s": 158, "fps": 30, "width": 1280, "height": 720}


def _seed(db):
    project = db.add_project("event")
    video = db.add_video(project, Path("Q1.mp4"), META, "EV", "Q1")
    robots = {team: db.add_robot(video, team, alliance, 10, 10)
              for team, alliance in ((1001, "red"), (1002, "red"), (2001, "blue"), (2002, "blue"))}
    return project, video, robots


def _confirm(db, robot_id, points, event_type="nectar_floor", recipient=None):
    event_id = db.add_scoring_event(robot_id, 100.0)
    db.review_event(event_id, "confirmed", points, robot_id, event_type, "high", recipient)
    return event_id


def _board(db, video):
    row = dict(db.connection.execute("SELECT * FROM videos WHERE id=?", (video,)).fetchone())
    return match_scoreboard(row, [dict(r) for r in db.robots_for_video(video)],
                            [dict(e) for e in db.events_for_video(video)])


class MatchRulingDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "project.sqlite"
        self.db = ProjectDatabase(self.path)
        self.addCleanup(lambda: self.db.close())
        self.project, self.video, self.robots = _seed(self.db)

    def reopen(self):
        self.db.close()
        self.db = ProjectDatabase(self.path)

    def test_new_video_defaults_to_qualification_without_ruling(self):
        self.assertEqual(self.db.match_ruling(self.video), {
            "match_type": "qualification", "qualification_disqualified_robot_ids": set(),
            "playoff_disqualified": set(), "first_disqualified": None,
            "simultaneous_disqualification": False, "match_final": False})

    def test_rulings_survive_reopen(self):
        self.db.set_match_ruling(self.video, "qualification",
                                 qualification_disqualified_robot_ids=[self.robots[1001]])
        self.reopen()
        self.assertEqual(self.db.match_ruling(self.video)["qualification_disqualified_robot_ids"],
                         {self.robots[1001]})
        self.db.set_match_type(self.video, "playoff", clear_inapplicable=True)
        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"},
                                 first_disqualified="blue", match_final=True)
        self.reopen()
        ruling = self.db.match_ruling(self.video)
        self.assertEqual((ruling["match_type"], ruling["playoff_disqualified"], ruling["first_disqualified"],
                          ruling["simultaneous_disqualification"], ruling["match_final"],
                          ruling["qualification_disqualified_robot_ids"]),
                         ("playoff", {"red", "blue"}, "blue", False, True, set()))
        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"},
                                 simultaneous_disqualification=True)
        self.reopen()
        ruling = self.db.match_ruling(self.video)
        self.assertEqual((ruling["first_disqualified"], ruling["simultaneous_disqualification"]), (None, True))

    def test_invalid_rulings_are_refused_and_nothing_is_written(self):
        other_video = self.db.add_video(self.project, Path("Q2.mp4"), META)
        other_robot = self.db.add_robot(other_video, 3001, "red", 1, 1)
        self.db.set_match_ruling(self.video, "qualification",
                                 qualification_disqualified_robot_ids=[self.robots[2001]])
        before = self.db.match_ruling(self.video)
        cases = [
            ("Match type", dict(match_type="final")),
            ("Unknown playoff DQ alliance", dict(match_type="playoff", playoff_disqualified={"green"})),
            ("Unknown first", dict(match_type="playoff", playoff_disqualified={"red", "blue"},
                                   first_disqualified="green")),
            ("not in this video", dict(match_type="qualification",
                                       qualification_disqualified_robot_ids=[other_robot])),
            ("needs both", dict(match_type="playoff", playoff_disqualified={"red"}, first_disqualified="red")),
            ("simultaneous", dict(match_type="playoff", playoff_disqualified={"red", "blue"},
                                  first_disqualified="red", simultaneous_disqualification=True)),
            ("simultaneous", dict(match_type="playoff", playoff_disqualified={"red"},
                                  simultaneous_disqualification=True)),
            ("cannot be combined", dict(match_type="playoff", playoff_disqualified={"red"},
                                        qualification_disqualified_robot_ids=[self.robots[1001]])),
            ("playoff matches only", dict(match_type="qualification", playoff_disqualified={"red"})),
            ("playoff matches only", dict(match_type="qualification", match_final=True)),
            ("team DQ is for qualification", dict(match_type="playoff",
                                                  qualification_disqualified_robot_ids=[self.robots[1001]])),
            ("integers", dict(match_type="qualification", qualification_disqualified_robot_ids=[True])),
        ]
        for message, kwargs in cases:
            with self.subTest(message=message, kwargs=kwargs):
                match_type = kwargs.pop("match_type")
                with self.assertRaisesRegex(ValueError, message):
                    self.db.set_match_ruling(self.video, match_type, **kwargs)
                self.assertEqual(self.db.match_ruling(self.video), before)
        with self.assertRaisesRegex(ValueError, "Unknown video"):
            self.db.set_match_ruling(999, "qualification")

    def test_mode_switch_refuses_or_clears_inapplicable_rulings(self):
        self.db.set_match_ruling(self.video, "qualification",
                                 qualification_disqualified_robot_ids=[self.robots[1001]])
        with self.assertRaisesRegex(ValueError, "qualification team DQ"):
            self.db.set_match_type(self.video, "playoff")
        self.assertEqual(self.db.match_ruling(self.video)["match_type"], "qualification")
        self.db.set_match_type(self.video, "playoff", clear_inapplicable=True)
        ruling = self.db.match_ruling(self.video)
        self.assertEqual((ruling["match_type"], ruling["qualification_disqualified_robot_ids"]), ("playoff", set()))

        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"},
                                 first_disqualified="red", match_final=True)
        with self.assertRaisesRegex(ValueError, "playoff alliance DQ"):
            self.db.set_match_type(self.video, "qualification")
        self.assertEqual(self.db.match_ruling(self.video)["playoff_disqualified"], {"red", "blue"})
        self.db.set_match_type(self.video, "qualification", clear_inapplicable=True)
        self.reopen()
        self.assertEqual(self.db.match_ruling(self.video), {
            "match_type": "qualification", "qualification_disqualified_robot_ids": set(),
            "playoff_disqualified": set(), "first_disqualified": None,
            "simultaneous_disqualification": False, "match_final": False})
        # Back to playoff: nothing from the earlier playoff ruling reappears.
        self.db.set_match_type(self.video, "playoff")
        self.assertEqual(self.db.match_ruling(self.video)["playoff_disqualified"], set())
        # A switch with nothing to clear is allowed without the flag.
        self.db.set_match_type(self.video, "qualification")
        with self.assertRaisesRegex(ValueError, "Match type"):
            self.db.set_match_type(self.video, "practice")

    def test_official_scores_keep_the_ruling_unless_an_alliance_dq_is_given(self):
        self.db.set_match_ruling(self.video, "qualification",
                                 qualification_disqualified_robot_ids=[self.robots[1001]])
        self.db.set_official_scores(self.video, 50, 40)
        self.assertEqual(self.db.match_ruling(self.video)["qualification_disqualified_robot_ids"],
                         {self.robots[1001]})
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.db.set_official_scores(self.video, 50, 40, {"red"})
        self.db.set_match_type(self.video, "playoff", clear_inapplicable=True)
        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"},
                                 first_disqualified="red")
        # Saving scores or thresholds alone never touches the playoff ruling.
        self.db.set_official_scores(self.video, 60, 45, swarm_threshold=18)
        ruling = self.db.match_ruling(self.video)
        self.assertEqual((ruling["playoff_disqualified"], ruling["first_disqualified"]), ({"red", "blue"}, "red"))
        self.db.set_official_scores(self.video, 50, 40, {"red"})
        ruling = self.db.match_ruling(self.video)
        self.assertEqual((ruling["playoff_disqualified"], ruling["first_disqualified"]), ({"red"}, None))


class LegacyMigrationTests(unittest.TestCase):
    def test_legacy_alliance_dq_becomes_a_playoff_ruling(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                "CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT);"
                "CREATE TABLE videos (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, path TEXT NOT NULL,"
                " duration_s REAL, fps REAL, width INTEGER, height INTEGER, event_code TEXT, match_number TEXT,"
                " official_red INTEGER, official_blue INTEGER, red_disqualified INTEGER NOT NULL DEFAULT 0,"
                " blue_disqualified INTEGER NOT NULL DEFAULT 0);"
                "CREATE TABLE robots (id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL, team_number INTEGER NOT NULL,"
                " alliance TEXT NOT NULL, color_hint TEXT);"
                "INSERT INTO projects VALUES (1, 'old', NULL);"
                "INSERT INTO videos (id, project_id, path, red_disqualified) VALUES (1, 1, 'q.mp4', 0), (2, 1, 'p.mp4', 1);"
                "INSERT INTO robots VALUES (1, 1, 1001, 'red', NULL), (2, 2, 2001, 'blue', NULL);")
            connection.commit()
            connection.close()
            for _ in range(2):  # the migration is idempotent
                db = ProjectDatabase(path)
                try:
                    self.assertEqual(db.match_ruling(1)["match_type"], "qualification")
                    playoff = db.match_ruling(2)
                    self.assertEqual((playoff["match_type"], playoff["playoff_disqualified"],
                                      playoff["first_disqualified"], playoff["match_final"]),
                                     ("playoff", {"red"}, None, False))
                    self.assertEqual([row["qualification_disqualified"] for row in
                                      db.connection.execute("SELECT * FROM robots ORDER BY id")], [0, 0])
                    board = _board(db, 2)
                    self.assertEqual(board["playoff_outcome"]["winner"], "blue")
                finally:
                    db.close()


class MatchScoringTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = ProjectDatabase(Path(folder.name) / "project.sqlite")
        self.addCleanup(lambda: self.db.close())
        _, self.video, self.robots = _seed(self.db)
        _confirm(self.db, self.robots[1001], 20)
        _confirm(self.db, self.robots[1002], 10)
        _confirm(self.db, self.robots[2001], 5)
        # A red major foul: the 20 points go to blue, never to the offending robot.
        _confirm(self.db, self.robots[1002], 20, "major_foul", "blue")

    def assertFoulInvariant(self, board):
        self.assertEqual(board["team_points"][self.robots[1002]], 0 if board["ruling"]["playoff_disqualified"]
                         & {"red"} or self.robots[1002] in board["ruling"]["qualification_disqualified_robot_ids"]
                         else 10)
        self.assertEqual(board["team_points"][self.robots[2001]], 0 if "blue" in board["ruling"]["playoff_disqualified"]
                         else 5)

    def test_qualification_without_dq(self):
        board = _board(self.db, self.video)
        self.assertEqual(board["match_type"], "qualification")
        self.assertEqual(board["alliance_points"], {"red": 30, "blue": 25})
        self.assertEqual(board["team_ranking_points"][self.robots[1001]], 3)
        self.assertFoulInvariant(board)

    def test_one_qualification_team_dq_leaves_partner_points_and_rp(self):
        self.db.set_match_ruling(self.video, "qualification",
                                 qualification_disqualified_robot_ids=[self.robots[1001]])
        board = _board(self.db, self.video)
        self.assertEqual(board["team_points"][self.robots[1001]], 0)
        self.assertEqual(board["team_ranking_points"][self.robots[1001]], 0)
        self.assertEqual(board["team_match_points"][self.robots[1001]], 0)
        self.assertEqual(board["team_points"][self.robots[1002]], 10)
        self.assertEqual(board["team_ranking_points"][self.robots[1002]], 3)
        self.assertEqual(board["team_match_points"][self.robots[1002]], 30)
        self.assertEqual(board["alliance_points"], {"red": 30, "blue": 25})
        self.assertEqual(board["ruling"]["playoff_disqualified"], set())
        self.assertFalse(board["pending_referee_ruling"])
        self.assertFoulInvariant(board)

    def test_both_qualification_teams_on_one_alliance_dq(self):
        self.db.set_match_ruling(self.video, "qualification",
                                 qualification_disqualified_robot_ids=[self.robots[1001], self.robots[1002]])
        board = _board(self.db, self.video)
        for team in (1001, 1002):
            self.assertEqual((board["team_points"][self.robots[team]], board["team_ranking_points"][self.robots[team]],
                              board["team_match_points"][self.robots[team]]), (0, 0, 0))
        # Team DQ is not an alliance DQ: the opponents are scored as usual.
        self.assertEqual(board["disqualified"], [])
        self.assertEqual(board["team_match_points"][self.robots[2001]], 25)
        self.assertEqual(board["team_points"][self.robots[2001]], 5)

    def test_playoff_single_dq(self):
        self.db.set_match_type(self.video, "playoff")
        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red"})
        board = _board(self.db, self.video)
        self.assertEqual(board["alliance_points"]["red"], 0)
        self.assertEqual(board["playoff_outcome"], {"resolved": True, "winner": "blue", "tie": False})
        self.assertEqual(set(board["team_ranking_points"].values()), {0})
        self.assertFalse(board["pending_referee_ruling"])
        self.assertFoulInvariant(board)

    def test_playoff_double_dq_rulings(self):
        self.db.set_match_type(self.video, "playoff")
        cases = (({}, {"resolved": False, "winner": None, "tie": False}, True),
                 ({"first_disqualified": "red"}, {"resolved": True, "winner": "blue", "tie": False}, False),
                 ({"first_disqualified": "blue"}, {"resolved": True, "winner": "red", "tie": False}, False),
                 ({"simultaneous_disqualification": True}, {"resolved": True, "winner": None, "tie": True}, False))
        for ruling, outcome, pending in cases:
            with self.subTest(ruling=ruling):
                self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"}, **ruling)
                board = _board(self.db, self.video)
                self.assertEqual(board["playoff_outcome"], outcome)
                self.assertEqual(board["pending_referee_ruling"], pending)
                self.assertEqual(board["alliance_points"], {"red": 0, "blue": 0})

    def test_playoff_without_dq_needs_match_final(self):
        self.db.set_match_type(self.video, "playoff")
        board = _board(self.db, self.video)
        self.assertEqual(board["playoff_outcome"]["resolved"], False)
        self.assertFalse(board["pending_referee_ruling"])
        self.db.set_match_ruling(self.video, "playoff", match_final=True)
        board = _board(self.db, self.video)
        self.assertEqual(board["playoff_outcome"], {"resolved": True, "winner": "red", "tie": False})
        self.assertEqual(set(board["team_ranking_points"].values()), {0})
        self.assertFoulInvariant(board)

    def test_inconsistent_rows_are_refused_not_mixed(self):
        robots = [dict(row) for row in self.db.robots_for_video(self.video)]
        with self.assertRaisesRegex(ValueError, "qualification match cannot carry"):
            match_scoreboard({"match_type": "qualification", "red_disqualified": 1}, robots, [])
        robots[0]["qualification_disqualified"] = 1
        with self.assertRaisesRegex(ValueError, "playoff match cannot carry"):
            match_scoreboard({"match_type": "playoff"}, robots, [])
        with self.assertRaisesRegex(ValueError, "Unknown match type"):
            match_scoreboard({"match_type": "final"}, robots, [])


class MatchExportTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.db = ProjectDatabase(self.folder / "project.sqlite")
        self.addCleanup(lambda: self.db.close())
        _, self.video, self.robots = _seed(self.db)
        _confirm(self.db, self.robots[1001], 30)
        _confirm(self.db, self.robots[2001], 5)
        foul = _confirm(self.db, self.robots[1002], 20, "major_foul", "blue")
        self.db.add_clip(foul, self.folder / "foul.mp4", 0.0, 6.0)

    def export(self):
        db, video = self.db, self.video
        row = db.connection.execute("SELECT * FROM videos WHERE id=?", (video,)).fetchone()
        target = self.folder / "scout.json"
        args = (db.robots_for_video(video), db.tracks_for_video(video), db.cycles_for_video(video))
        export_scout_json(target, row, *args, db.events_for_video(video), db.clips_for_video(video),
                          db.photos_for_video(video), {})
        csv_path = self.folder / "stats.csv"
        export_stats_csv(csv_path, *args, {}, events=db.events_for_video(video), video=row)
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        return json.loads(target.read_text(encoding="utf-8")), csv_rows

    def test_qualification_export_has_team_dq_and_rp(self):
        self.db.set_match_ruling(self.video, "qualification",
                                 qualification_disqualified_robot_ids=[self.robots[1001]])
        payload, csv_rows = self.export()
        for key in ("app", "version", "event_code", "match", "video", "teams", "clips", "photos"):
            self.assertIn(key, payload)
        self.assertEqual(payload["version"], "1.0.0")
        self.assertEqual(payload["match_type"], "qualification")
        self.assertEqual(payload["ruling"], {"qualification_disqualified_teams": [1001], "alliance_disqualified": [],
                                             "first_disqualified": None, "simultaneous_disqualification": False,
                                             "match_final": False})
        self.assertIsNone(payload["playoff_outcome"])
        teams = {team["team_number"]: team for team in payload["teams"]}
        self.assertEqual((teams[1001]["disqualified"], teams[1001]["points_video"],
                          teams[1001]["ranking_points_video"]), (True, 0, 0))
        self.assertEqual((teams[1002]["disqualified"], teams[1002]["ranking_points_video"]), (False, 3))
        self.assertEqual(teams[1002]["points_video"], 0)  # the foul is never credited to the offender
        self.assertIn("WIN_TIE", payload["alliance_results"]["red"])
        by_team = {int(row["team_number"]): row for row in csv_rows}
        self.assertEqual((by_team[1001]["match_type"], by_team[1001]["disqualified"],
                          by_team[1001]["ranking_points_video"], by_team[1001]["playoff_result"]),
                         ("qualification", "1", "0", ""))
        self.assertEqual(by_team[1002]["ranking_points_video"], "3")
        foul = payload["clips"][0]
        self.assertEqual((foul["recipient_alliance"], foul["points"], foul["robot_points"]), ("blue", 20, 0))

    def test_playoff_export_has_outcome_and_no_rp(self):
        self.db.set_match_type(self.video, "playoff")
        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"})
        payload, csv_rows = self.export()
        self.assertEqual(payload["match_type"], "playoff")
        self.assertEqual(payload["playoff_outcome"], {"resolved": False, "winner": None, "tie": False,
                                                      "pending_referee_ruling": True})
        self.assertTrue(all(team["ranking_points_video"] is None for team in payload["teams"]))
        self.assertTrue(all(team["disqualified"] for team in payload["teams"]))
        self.assertEqual(set(payload["alliance_results"]["red"]), {"points_video"})
        self.assertEqual({row["playoff_result"] for row in csv_rows}, {"pending"})
        self.assertEqual({row["ranking_points_video"] for row in csv_rows}, {""})

        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"},
                                 first_disqualified="red")
        payload, csv_rows = self.export()
        self.assertEqual(payload["playoff_outcome"], {"resolved": True, "winner": "blue", "tie": False,
                                                      "pending_referee_ruling": False})
        self.assertEqual(payload["ruling"]["first_disqualified"], "red")
        self.assertEqual({int(row["team_number"]): row["playoff_result"] for row in csv_rows},
                         {1001: "loss", 1002: "loss", 2001: "win", 2002: "win"})

        self.db.set_match_ruling(self.video, "playoff", match_final=True)
        payload, csv_rows = self.export()
        # Red 30 vs blue 5 + 20 foul credit: the reviewed final score decides.
        self.assertEqual(payload["playoff_outcome"]["winner"], "red")
        self.assertEqual(payload["alliance_results"], {"red": {"points_video": 30}, "blue": {"points_video": 25}})
        self.assertFalse(any(team["disqualified"] for team in payload["teams"]))


class MatchRulingUiTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from control_vision.app.main_window import MainWindow
        self.app = QApplication.instance() or QApplication([])
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = ProjectDatabase(Path(folder.name) / "project.sqlite")
        self.project, self.video, self.robots = _seed(self.db)
        _confirm(self.db, self.robots[1001], 20)
        self.db.set_analysis_status(self.video, "ready")
        # A modal warning would block an offscreen run; record it instead.
        warning = patch("control_vision.app.main_window.QMessageBox.warning")
        self.warning = warning.start()
        self.addCleanup(warning.stop)
        self.window = MainWindow()
        self.window.db, self.window.project_id, self.window.video_id = self.db, self.project, self.video
        self.window.load_match_ruling()
        self.window.update_score_comparison()

    def tearDown(self):
        self.window.db = None
        self.window.close()
        self.db.close()
        # Every action in these tests is valid, so nothing is refused.
        self.warning.assert_not_called()

    def switch(self, match_type, answer):
        from PySide6.QtWidgets import QMessageBox
        button = QMessageBox.StandardButton.Yes if answer else QMessageBox.StandardButton.No
        with patch("control_vision.app.main_window.QMessageBox.question", return_value=button) as asked:
            self.window.match_type_combo.setCurrentIndex(self.window.match_type_combo.findData(match_type))
        return asked

    def test_qualification_shows_team_dq_per_seeded_robot(self):
        window = self.window
        self.assertFalse(window.qualification_panel.isHidden())
        self.assertTrue(window.playoff_panel.isHidden())
        self.assertEqual(sorted(window.team_dq_checks), sorted(self.robots.values()))
        label = window.team_dq_checks[self.robots[1001]].text()
        self.assertIn("1001", label)
        self.assertTrue(window.team_dq_checks[self.robots[1001]].accessibleName())
        window.team_dq_checks[self.robots[1001]].click()
        self.assertEqual(self.db.match_ruling(self.video)["qualification_disqualified_robot_ids"],
                         {self.robots[1001]})
        self.assertIn("1001", window.ruling_status.text())
        self.assertIn("T601", window.ruling_status.text())

    def test_mode_switching_asks_before_clearing_and_shows_pending_ruling(self):
        window = self.window
        window.team_dq_checks[self.robots[1002]].click()
        asked = self.switch("playoff", answer=False)
        asked.assert_called_once()
        self.assertEqual(window.match_type_combo.currentData(), "qualification")
        self.assertEqual(self.db.match_ruling(self.video)["qualification_disqualified_robot_ids"],
                         {self.robots[1002]})
        self.assertFalse(window.qualification_panel.isHidden())

        self.switch("playoff", answer=True)
        ruling = self.db.match_ruling(self.video)
        self.assertEqual((ruling["match_type"], ruling["qualification_disqualified_robot_ids"]), ("playoff", set()))
        self.assertTrue(window.qualification_panel.isHidden())
        self.assertFalse(window.playoff_panel.isHidden())
        self.assertFalse(window.ruling_combo.isEnabled())
        self.assertIn("do not earn ranking points", window.ranking_summary.text().lower())

        window.disq_red.click()
        self.assertIn("Blue wins", window.ruling_status.text())
        window.disq_blue.click()
        self.assertTrue(window.ruling_combo.isEnabled())
        self.assertIn("PENDING REFEREE RULING", window.ruling_status.text())
        self.assertEqual(self.db.match_ruling(self.video)["playoff_disqualified"], {"red", "blue"})
        window.ruling_combo.setCurrentIndex(window.ruling_combo.findData("red"))
        self.assertEqual(self.db.match_ruling(self.video)["first_disqualified"], "red")
        self.assertIn("Blue wins", window.ruling_status.text())
        window.ruling_combo.setCurrentIndex(window.ruling_combo.findData("simultaneous"))
        self.assertIn("tie", window.ruling_status.text())
        self.assertTrue(self.db.match_ruling(self.video)["simultaneous_disqualification"])
        # Un-DQ one alliance: the double-DQ ruling no longer applies and is dropped.
        window.disq_blue.click()
        ruling = self.db.match_ruling(self.video)
        self.assertEqual((ruling["playoff_disqualified"], ruling["simultaneous_disqualification"]),
                         ({"red"}, False))
        self.assertEqual(window.ruling_combo.currentIndex(), 0)
        self.assertFalse(window.ruling_combo.isEnabled())

        self.switch("qualification", answer=True)
        self.assertEqual(self.db.match_ruling(self.video), {
            "match_type": "qualification", "qualification_disqualified_robot_ids": set(),
            "playoff_disqualified": set(), "first_disqualified": None,
            "simultaneous_disqualification": False, "match_final": False})
        self.assertFalse(window.disq_red.isChecked())

    def test_reloading_the_video_restores_the_stored_ruling(self):
        self.db.set_match_type(self.video, "playoff")
        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"})
        self.window.load_match_ruling()
        self.window.update_score_comparison()
        self.assertEqual(self.window.match_type_combo.currentData(), "playoff")
        self.assertTrue(self.window.disq_red.isChecked() and self.window.disq_blue.isChecked())
        self.assertIn("PENDING REFEREE RULING", self.window.ruling_status.text())
        self.window.video_id = None
        self.window.load_match_ruling()
        self.assertEqual(self.window.match_type_combo.currentData(), "qualification")
        self.assertFalse(self.window.disq_red.isChecked())
        self.assertEqual(self.window.team_dq_checks, {})


if __name__ == "__main__":
    unittest.main()
