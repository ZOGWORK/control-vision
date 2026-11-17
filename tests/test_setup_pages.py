"""Import, Livestream splitter, Calibrate and Robots pages (UI report pages 4-7)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.storage.database import ProjectDatabase

META = {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720}
CORNERS = [[60.0, 40.0], [260.0, 40.0], [300.0, 200.0], [20.0, 200.0]]


def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def write_video(path: Path, seconds: float = 2.0, fps: int = 10, size=(320, 240)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(int(seconds * fps)):
        frame = np.full((size[1], size[0], 3), (index * 9) % 255, np.uint8)
        cv2.rectangle(frame, (20 + index, 40), (60 + index, 80), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class SetupTest(unittest.TestCase):
    def setUp(self):
        self.app = qt_app()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.db = ProjectDatabase(self.root / "control_vision.sqlite")
        self.addCleanup(self.db.close)
        self.project = self.db.add_project("Meet")

    def add_video(self, name="Q1.mp4", real=True, **meta):
        path = write_video(self.root / name) if real else self.root / name
        metadata = dict(META, **meta)
        if real:
            metadata.update(duration_s=2.0, fps=10.0, width=320, height=240)
        return self.db.add_video(self.project, path, metadata, None, None), path


class ClockAndFileTests(SetupTest):
    def test_clock_text_round_trips(self):
        from control_vision.app.media import format_clock, parse_clock
        self.assertEqual(format_clock(5025), "01:23:45")
        self.assertEqual(format_clock(150), "02:30")
        self.assertEqual(parse_clock("1:23:45"), 5025)
        self.assertEqual(parse_clock("2:30"), 150)
        self.assertEqual(parse_clock("45.5"), 45.5)
        for bad in ("", "1:2:3:4", "1:75", "abc", "-5"):
            with self.assertRaises(ValueError):
                parse_clock(bad)

    def test_file_checks_read_the_file(self):
        from control_vision.app.media import check_file
        unsupported = self.root / "notes.txt"
        unsupported.write_text("x")
        self.assertIn("not supported", check_file(unsupported).error)
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"not a video")
        self.assertIsNotNone(check_file(broken).error)
        good = check_file(write_video(self.root / "good.mp4"))
        self.assertTrue(good.ok, good.error)
        self.assertEqual(good.resolution, "320x240")
        self.assertAlmostEqual(good.duration_s, 2.0, delta=0.2)
        self.assertFalse(good.vertical)
        tall = check_file(write_video(self.root / "tall.mp4", size=(240, 320)))
        self.assertTrue(tall.vertical)
        self.assertIn("Vertical", " ".join(tall.warnings))

    def test_cut_clip_writes_a_new_file_and_never_overwrites(self):
        from control_vision.app.media import cut_clip
        from control_vision.pipeline.ingest import probe_video
        source = write_video(self.root / "stream.mp4", seconds=4)
        out = cut_clip(source, 1.0, 3.0, self.root / "matches" / "m1.mp4")
        self.assertTrue(out.is_file())
        self.assertLess(probe_video(out)["duration_s"], 3.6)
        with self.assertRaises(FileExistsError):
            cut_clip(source, 1.0, 3.0, out)
        with self.assertRaises(ValueError):
            cut_clip(source, 3.0, 1.0, self.root / "m2.mp4")


class IdentityStorageTests(SetupTest):
    def test_identity_updates_in_place_and_keeps_results_linked(self):
        video, _ = self.add_video(real=False)
        first = self.db.save_identity(video, [
            {"team_number": 21229, "alliance": "red", "start_position": "left", "absent": False, "seed_x": 10, "seed_y": 20},
            {"team_number": 16655, "alliance": "blue", "start_position": None, "absent": True}])
        event = self.db.add_scoring_event(first[0], 12.0)
        again = self.db.save_identity(video, [
            {"id": first[0], "team_number": 21230, "alliance": "red", "start_position": "right", "absent": False,
             "seed_x": 11, "seed_y": 21}])
        self.assertEqual(again, [first[0]])
        robots = self.db.robots_for_video(video)
        self.assertEqual([(r["team_number"], r["start_position"], r["absent"]) for r in robots], [(21230, "right", 0)])
        self.assertEqual(self.db.events_for_video(video)[0]["robot_id"], first[0])
        self.assertEqual(self.db.events_for_video(video)[0]["id"], event)

    def test_identity_refuses_duplicates_and_missing_clicks(self):
        video, _ = self.add_video(real=False)
        with self.assertRaises(ValueError):
            self.db.save_identity(video, [{"team_number": 1, "alliance": "red", "seed_x": 1, "seed_y": 1},
                                          {"team_number": 1, "alliance": "blue", "seed_x": 2, "seed_y": 2}])
        with self.assertRaises(ValueError):
            self.db.save_identity(video, [{"team_number": 1, "alliance": "red", "absent": False}])
        with self.assertRaises(ValueError):
            self.db.save_identity(video, [{"team_number": 1, "alliance": "green", "seed_x": 1, "seed_y": 1}])
        self.assertEqual(list(self.db.robots_for_video(video)), [])


class ImportPageTests(SetupTest):
    def page(self):
        from control_vision.app.media import FileChecker
        from control_vision.app.pages.import_page import ImportPage
        page = ImportPage(FileChecker(synchronous=True))
        page.set_project(self.db, self.project)
        return page

    def test_import_adds_paths_with_holdout_and_camera_view(self):
        page = self.page()
        good = write_video(self.root / "Q2.mp4")
        bad = self.root / "notes.txt"; bad.write_text("x")
        page.add_paths([good, bad])
        self.assertEqual(len(page.ready()), 1)
        self.assertIn("not supported", page.table.item(1, 8).text())
        page.queue[0].holdout, page.queue[0].camera_view, page.queue[0].match = True, "wide", "Q2"
        with patch.object(page, "imported") as signal:
            ids = page.import_ready()
        signal.emit.assert_called_once_with(ids)
        row = self.db.connection.execute("SELECT * FROM videos WHERE id=?", (ids[0],)).fetchone()
        self.assertEqual((row["is_holdout"], row["camera_view"], row["match_number"], Path(row["path"])),
                         (1, "wide", "Q2", good))
        self.assertTrue(good.is_file())  # The file is never copied or moved.
        self.assertEqual(len(page.queue), 1)
        self.assertIn("Needs calibration", page.project_table.item(0, 6).text())
        self.assertIn("Holdout", page.project_table.item(0, 5).text())

    def test_duplicates_are_flagged_not_dropped(self):
        page = self.page()
        _, path = self.add_video("Q1.mp4")
        page.add_paths([path])
        self.assertEqual(page.duplicates(), [True])
        self.assertEqual(page.ready(), [])
        self.assertIn("Duplicate", page.table.item(0, 8).text())
        page.table.selectRow(0)
        page.import_anyway()
        self.assertEqual(len(page.ready()), 1)

    def test_camera_view_and_holdout_can_be_changed_after_import(self):
        from PySide6.QtCore import Qt
        page = self.page()
        video, _ = self.add_video()
        page.refresh_project()
        page.project_table.item(0, 5).setCheckState(Qt.CheckState.Checked)
        combo = page.project_table.cellWidget(0, 4)
        combo.setCurrentIndex(combo.findData("close"))
        row = self.db.connection.execute("SELECT is_holdout,camera_view FROM videos WHERE id=?", (video,)).fetchone()
        self.assertEqual((row["is_holdout"], row["camera_view"]), (1, "close"))

    def test_refresh_keeps_cell_widgets_and_typed_text(self):
        page = self.page()
        first, second = write_video(self.root / "Q4.mp4"), write_video(self.root / "Q5.mp4")
        page.add_paths([first, second])
        match = page.table.cellWidget(1, 6)
        match.setText("Q5")
        holdout = page.table.cellWidget(1, 7)
        page._render_queue()  # e.g. another file's check finished
        self.assertIs(page.table.cellWidget(1, 6), match)
        self.assertIs(page.table.cellWidget(1, 7), holdout)
        self.assertEqual(match.text(), "Q5")
        page.table.selectRow(0)
        page.remove_selected()
        # The remaining file's widgets move up with it; the removed file's are not left on screen.
        self.assertEqual(page.table.cellWidget(0, 6).text(), "Q5")
        self.assertEqual(page.table.cellWidget(0, 6).property("cell_key"), str(second.resolve()))
        video, _ = self.add_video()
        page.refresh_project()
        combo = page.project_table.cellWidget(0, 4)
        page.refresh_project()
        self.assertIs(page.project_table.cellWidget(0, 4), combo)

    def test_youtube_links_are_never_downloaded(self):
        page = self.page()
        page.youtube.setText("https://youtu.be/abcdefghijk")
        page.check_link()
        self.assertIn("not available in this build", page.youtube_status.text())
        self.assertEqual(page.queue, [])


class SplitterTests(SetupTest):
    def test_split_editing_rules(self):
        from control_vision.app.pages.splitter import Segment, add_segment, adjust_segment, merge_segments
        segments = add_segment([], 100, 250, 1000)
        segments = add_segment(segments, 300, 450, 1000)
        with self.assertRaises(ValueError):
            add_segment(segments, 200, 320, 1000)
        with self.assertRaises(ValueError):
            add_segment(segments, 500, 500.5, 1000)
        segments[0].confirmed = True
        adjust_segment(segments, 0, end=400, duration=1000)
        self.assertEqual(segments[0].end, 300)  # Stops at the next match.
        self.assertFalse(segments[0].confirmed)  # A moved split needs checking again.
        segments[0].match, segments[1].match = "Q4", "Q4"
        merged = merge_segments(segments, 0)
        self.assertEqual((merged[0].start, merged[0].end, merged[0].match), (100, 450, "Q4"))
        self.assertTrue(Segment(0, 30).short)
        with self.assertRaises(ValueError):
            merge_segments(merged, 0)

    def test_draft_survives_reopening(self):
        from control_vision.app.pages.splitter import Segment, load_draft, save_draft
        source = self.root / "stream.mp4"
        save_draft(self.root, source, [Segment(10, 170, "Q1", True)])
        self.assertEqual(load_draft(self.root, source), [Segment(10, 170, "Q1", True)])
        (self.root / "livestream_splits.json").write_text("not json", encoding="utf-8")
        self.assertEqual(load_draft(self.root, source), [])

    def test_confirmed_matches_become_project_videos(self):
        from control_vision.app.pages import splitter
        source = write_video(self.root / "stream.mp4", seconds=6)
        page = splitter.SplitterPage()
        self.addCleanup(page.close_source)
        page.set_project(self.db, self.project, self.root)
        self.assertTrue(page.open_source(source))
        self.assertFalse(page.create_btn.isEnabled())
        page.start_edit.setText("0:01"); page.end_edit.setText("0:03")
        page.add_typed()
        page.timeline.mark_in = 3.5
        page.seek(5.5)
        page.mark_out()
        self.assertEqual(len(page.segments), 2)
        self.assertIn("Unknown match", page.cards.item(0).text())
        self.assertIn("Very short", page.cards.item(0).text())
        page.cards.setCurrentRow(0)
        page.match_edit.setText("Q7"); page._set_match_number()
        page.confirm_selected()
        with patch.object(splitter.ClipWorker, "start", lambda worker: worker.run()), \
                patch.object(page, "clips_created") as created:
            page.create_clips()
        ids = created.emit.call_args.args[0]
        row = self.db.connection.execute("SELECT path,match_number FROM videos WHERE id=?", (ids[0],)).fetchone()
        self.assertEqual(row["match_number"], "Q7")
        self.assertTrue(Path(row["path"]).is_file())
        self.assertEqual(Path(row["path"]).parent, self.root / "matches")
        self.assertEqual(len(page.segments), 1)  # The unconfirmed match stays to be checked.
        self.assertTrue(source.is_file())

    def test_detection_says_it_is_not_available(self):
        from control_vision.app.pages.splitter import SplitterPage
        page = SplitterPage()
        self.assertFalse(page.detect_btn.isEnabled())
        self.assertIn("not available", page.detect_btn.toolTip())


class CalibrateTests(SetupTest):
    def test_status_is_honest_about_the_corner_fit(self):
        from control_vision.app.pages.calibrate import calibration_status, side_view
        self.assertEqual(calibration_status([])[0], "Click corner 1 (top-left).")
        self.assertIn("corner 3", calibration_status(CORNERS[:2])[0])
        message, tone, savable = calibration_status(CORNERS)
        self.assertTrue(savable)
        self.assertIn("not an accuracy measure", message)
        reversed_corners = list(reversed(CORNERS))
        message, tone, savable = calibration_status(reversed_corners)
        self.assertEqual((tone, savable), ("danger", False))
        self.assertIn("Re-click corners", message)
        self.assertTrue(side_view([[140, 40], [180, 40], [300, 200], [20, 200]]))
        self.assertFalse(side_view([[0, 0], [100, 0], [100, 100], [0, 100]]))

    def test_saving_and_reusing_a_calibration(self):
        from control_vision.app.pages.calibrate import CalibratePage
        first, _ = self.add_video("Q1.mp4")
        second = self.db.add_video(self.project, self.root / "Q2.mp4",
                                   dict(META, width=640, height=480, duration_s=2.0), None, None)
        page = CalibratePage()
        page.set_project(self.db, self.project)
        self.assertEqual(int(page.video["id"]), first)
        self.assertFalse(page.save_btn.isEnabled())
        page.canvas.corners = [list(point) for point in CORNERS]
        page._corners_changed()
        page.add_zone("Red Loading Zone", "pickup", [[9, 0], [12, 0], [12, 3], [9, 3]])
        self.assertIsNotNone(page.map.background)  # The frame is shown warped onto the field.
        self.assertTrue(page.save())
        corners, zones = self.db.calibration_for(first)
        self.assertEqual(corners, CORNERS)
        self.assertEqual(zones["Red Loading Zone"]["type"], "pickup")
        self.assertEqual(page.apply_to([second]), 1)
        scaled, copied = self.db.calibration_for(second)
        self.assertEqual(scaled[0], [120.0, 80.0])  # 640x480 is twice 320x240.
        self.assertEqual(copied["Red Loading Zone"]["points"][0], [9, 0])  # Zones are in feet and stay.

    def test_leaving_unsaved_corners_asks_first(self):
        from control_vision.app.pages import calibrate
        self.add_video("Q1.mp4")
        page = calibrate.CalibratePage()
        page.set_project(self.db, self.project)
        page.canvas.corners = [list(point) for point in CORNERS]
        page._corners_changed()
        with patch.object(calibrate.QMessageBox, "question", return_value=calibrate.QMessageBox.StandardButton.Cancel):
            self.assertFalse(page.confirm_discard())
        with patch.object(calibrate.QMessageBox, "question", return_value=calibrate.QMessageBox.StandardButton.Save):
            self.assertTrue(page.confirm_discard())
        self.assertIsNotNone(self.db.calibration_for(int(page.video["id"])))

    def test_zone_names_are_checked(self):
        from control_vision.app.pages.calibrate import ZoneDialog, zone_type_for
        dialog = ZoneDialog({"Hive"})
        self.assertIsNotNone(dialog.problem_with("Hive"))
        self.assertIsNotNone(dialog.problem_with("goal_roi"))
        self.assertIsNone(dialog.problem_with("Flower 1"))
        self.assertEqual(zone_type_for("Blue Loading Zone"), "pickup")
        self.assertEqual(zone_type_for("Park"), "park")
        self.assertEqual(zone_type_for("Red Garden"), "scoring")


class IdentityTests(SetupTest):
    def test_states_follow_the_guide(self):
        from control_vision.app.pages.identity import RobotEntry, identity_state
        def robot(team, alliance, absent=False):
            return RobotEntry(team=team, alliance=alliance, absent=absent, seed=None if absent else (5, 5))
        four = [robot("1", "red"), robot("2", "red"), robot("3", "blue"), robot("4", "blue")]
        self.assertEqual(identity_state(four)[:2], ("✓ Ready to process.", "success"))
        message, tone, savable = identity_state([*four[:3], robot("4", "blue", absent=True)])
        self.assertIn("1 robot not visible", message)
        self.assertTrue(savable)
        message, tone, savable = identity_state([*four[:3], robot("3", "blue")])
        self.assertEqual((tone, savable), ("danger", False))
        self.assertIn("already assigned", message)
        message, _tone, savable = identity_state([*four[:3], robot("4", "red")])
        self.assertIn("3 Red, 1 Blue", message)
        self.assertTrue(savable)
        message, _tone, savable = identity_state([RobotEntry(team="9", seed=(1, 1)), RobotEntry(), RobotEntry(), RobotEntry()])
        self.assertIn("choose Red or Blue", message)
        self.assertFalse(savable)

    def test_page_saves_clicked_robots(self):
        from control_vision.app.pages.identity import IdentityPage
        video, _ = self.add_video()
        page = IdentityPage()
        page.set_project(self.db, self.project)
        self.assertFalse(page.next_btn.isEnabled())
        page.select(0)
        page._frame_clicked(40, 60)
        card = page.cards[0]
        card.team.setText("21229"); card.red.setChecked(True)
        card.position.setCurrentIndex(card.position.findData("left"))
        page._card_changed()
        page.cards[1].team.setText("16655"); page.cards[1].blue.setChecked(True); page.cards[1].absent.setChecked(True)
        page._card_changed()
        self.assertTrue(page.save_btn.isEnabled())
        self.assertTrue(page.save())
        robots = self.db.robots_for_video(video)
        self.assertEqual([(r["team_number"], r["alliance"], r["seed_x"], r["absent"], r["start_position"]) for r in robots],
                         [(21229, "red", 40, 0, "left"), (16655, "blue", None, 1, None)])
        self.assertTrue(page.next_btn.isEnabled())
        page.swap(0, 1)
        self.assertEqual((page.entries[0].team, page.entries[0].seed), ("16655", (40, 60)))

    def test_changing_alliances_after_processing_asks_and_hides_results(self):
        from control_vision.app.pages import identity
        video, _ = self.add_video()
        ids = self.db.save_identity(video, [{"team_number": 7, "alliance": "red", "seed_x": 5, "seed_y": 5}])
        self.db.set_analysis_status(video, "ready")
        page = identity.IdentityPage()
        page.set_project(self.db, self.project)
        page.cards[0].team.setText("8")
        page._card_changed()
        with patch.object(identity.QMessageBox, "question") as asked:
            self.assertTrue(page.save())  # A team number fix keeps the results.
        asked.assert_not_called()
        page.cards[0].blue.setChecked(True)
        page._card_changed()
        with patch.object(identity.QMessageBox, "question", return_value=identity.QMessageBox.StandardButton.Save):
            self.assertTrue(page.save())
        status = self.db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (video,)).fetchone()[0]
        self.assertEqual(status, "unprocessed")
        self.assertEqual([r["id"] for r in self.db.robots_for_video(video)], ids)


class WindowSetupTests(SetupTest):
    def setUp(self):
        super().setUp()
        from control_vision.app import library
        from control_vision.app.main_window import MainWindow
        self.window = MainWindow()
        self.window.library = library.ProjectLibrary(self.root / "home")
        self.window.model_store = library.ModelStore(self.root / "home")

    def tearDown(self):
        self.window.db = None
        self.window.close()

    def test_sidebar_lists_the_setup_pages_and_they_get_the_project(self):
        from control_vision.app import main_window as mw
        window = self.window
        for tab, label in ((mw.IMPORT_TAB, "Import videos"), (mw.SPLITTER_TAB, "Livestream splitter"),
                           (mw.CALIBRATE_TAB, "Calibrate field"), (mw.IDENTITY_TAB, "Robots")):
            self.assertEqual(window._nav_labels[tab], label)
        self.assertTrue(window.import_page.body.isHidden())
        video, _ = self.add_video()
        window.db, window.project_id = self.db, self.project
        window.refresh_home()
        self.assertFalse(window.import_page.body.isHidden())
        self.assertEqual(int(window.calibrate_page.video["id"]), video)

    def test_saving_a_calibration_updates_the_loaded_video(self):
        window = self.window
        video, _ = self.add_video()
        window.db, window.project_id = self.db, self.project
        window.refresh_home()
        window.load_saved_video_id(video)
        self.assertEqual(window.corners, [])
        page = window.calibrate_page
        page.canvas.corners = [list(point) for point in CORNERS]
        page._corners_changed()
        page.save()
        self.assertEqual(window.corners, CORNERS)
        self.assertEqual(window._step_widgets["calibrate"][0].property("state"), "done")


if __name__ == "__main__":
    unittest.main()
