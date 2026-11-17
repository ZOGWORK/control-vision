"""Import, Livestream splitter, Calibrate and Robots pages (UI report pages 4-7)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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


def select_row(table, row: int) -> None:
    """Select one row whatever keyboard modifiers an earlier test left behind (selectRow reads them)."""
    from PySide6.QtCore import QItemSelectionModel
    flags = QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
    table.selectionModel().select(table.model().index(row, 0), flags)


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
        for bad in ("", "1:2:3:4", "1:75", "abc", "-5", "nan", "1:nan", "inf", "1:00:inf"):
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

    def test_identity_and_reprocess_flag_are_one_transaction(self):
        video, _ = self.add_video()
        self.db.save_identity(video, [{"team_number": 7, "alliance": "red", "seed_x": 5, "seed_y": 5}])
        self.db.set_analysis_status(video, "ready")
        status = lambda: self.db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (video,)).fetchone()[0]
        with self.assertRaises(ValueError):  # Fails part-way: neither the robots nor the status change.
            self.db.save_identity(video, [{"team_number": 7, "alliance": "blue", "seed_x": 5, "seed_y": 5},
                                          {"team_number": 8, "alliance": "green", "seed_x": 1, "seed_y": 1}],
                                  mark_unprocessed=True)
        self.assertEqual((status(), self.db.robots_for_video(video)[0]["alliance"]), ("ready", "red"))
        self.db.save_identity(video, [{"team_number": 7, "alliance": "blue", "seed_x": 5, "seed_y": 5}], mark_unprocessed=True)
        self.assertEqual((status(), self.db.robots_for_video(video)[0]["alliance"]), ("unprocessed", "blue"))

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
        select_row(page.table, 0)
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
        select_row(page.table, 0)
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
        # The remaining match gets its own name, not the first one's (names follow the start time).
        page.cards.setCurrentRow(0)
        page.confirm_selected()
        with patch.object(splitter.ClipWorker, "start", lambda worker: worker.run()),                 patch.object(page, "clips_created") as created:
            page.create_clips()
        created.emit.assert_called_once()
        self.assertEqual(len(list((self.root / "matches").glob("*.mp4"))), 2)
        self.assertNotEqual(splitter.clip_name(source, splitter.Segment(1, 3)), splitter.clip_name(source, splitter.Segment(3.5, 5.5)))
        self.assertEqual(splitter.clip_name(source, splitter.Segment(65, 90, "Q 7")), "stream-at01-05-Q7.mp4")

    def test_clips_stay_with_the_project_they_were_cut_for(self):
        from control_vision.app.pages import splitter
        source = write_video(self.root / "stream.mp4", seconds=4)
        page = splitter.SplitterPage()
        self.addCleanup(page.close_source)
        page.set_project(self.db, self.project, self.root)
        page.open_source(source)
        page.start_edit.setText("0:01"); page.end_edit.setText("0:03")
        page.add_typed()
        page.cards.setCurrentRow(0)
        page.confirm_selected()
        with patch.object(splitter.ClipWorker, "start", lambda worker: None):
            page.create_clips()
        other = self.db.add_project("Other meet")
        page.set_project(self.db, other, self.root)  # Switched while the clip is being cut.
        with patch.object(page, "clips_created") as created:
            page.worker.run()
        created.emit.assert_not_called()
        projects = [row[0] for row in self.db.connection.execute("SELECT project_id FROM videos")]
        self.assertEqual(projects, [self.project])
        self.assertIsNone(page.worker)

    def test_deleting_a_split_shows_the_next_ones_number(self):
        from control_vision.app.pages.splitter import SplitterPage
        page = SplitterPage()
        self.addCleanup(page.close_source)
        page.set_project(self.db, self.project, self.root)
        page.open_source(write_video(self.root / "stream.mp4", seconds=6))
        for start, end, match in (("0:00", "0:02", "Q1"), ("0:03", "0:05", "Q2")):
            page.start_edit.setText(start); page.end_edit.setText(end)
            page.add_typed()
            page.cards.setCurrentRow(len(page.segments) - 1)
            page.match_edit.setText(match); page._set_match_number()
        page.cards.setCurrentRow(0)
        self.assertEqual(page.match_edit.text(), "Q1")
        page.delete_selected()
        self.assertEqual([segment.match for segment in page.segments], ["Q2"])
        self.assertEqual(page.match_edit.text(), "Q2")   # Not the deleted split's number.
        page._set_match_number()
        self.assertEqual(page.segments[0].match, "Q2")

    def test_closing_a_playing_stream_resets_play(self):
        from control_vision.app.pages.splitter import SplitterPage
        page = SplitterPage()
        self.addCleanup(page.close_source)
        page.set_project(self.db, self.project, self.root)
        page.open_source(write_video(self.root / "stream.mp4", seconds=3))
        page.toggle_play()
        self.assertEqual(page.play_btn.text(), "Pause")
        page.close_source()
        self.assertEqual(page.play_btn.text(), "Play")
        self.assertFalse(page.timer.isActive())

    def splitter_with_a_confirmed_match(self):
        from control_vision.app.pages import splitter
        source = write_video(self.root / "stream.mp4", seconds=4)
        page = splitter.SplitterPage()
        self.addCleanup(page.close_source)
        page.set_project(self.db, self.project, self.root)
        page.open_source(source)
        page.start_edit.setText("0:01"); page.end_edit.setText("0:03")
        page.add_typed()
        page.cards.setCurrentRow(0)
        page.confirm_selected()
        return splitter, page

    def test_a_clip_that_cannot_be_read_is_removed(self):
        splitter, page = self.splitter_with_a_confirmed_match()
        with patch.object(splitter.ClipWorker, "start", lambda worker: worker.run()), \
                patch("control_vision.pipeline.ingest.probe_video", side_effect=ValueError("unreadable")):
            page.create_clips()
        self.assertEqual(list((self.root / "matches").glob("*.mp4")), [])  # A retry is not blocked.
        self.assertIsNone(page.worker)
        with patch.object(splitter.ClipWorker, "start", lambda worker: worker.run()):
            page.create_clips()
        self.assertEqual(len(list((self.root / "matches").glob("*.mp4"))), 1)

    def test_a_database_error_still_finishes_the_run(self):
        import sqlite3
        splitter, page = self.splitter_with_a_confirmed_match()
        with patch.object(splitter.ClipWorker, "start", lambda worker: worker.run()), \
                patch.object(self.db, "add_video", side_effect=sqlite3.OperationalError("database is locked")):
            page.create_clips()
        self.assertIsNone(page.worker)
        self.assertTrue(page.create_btn.isEnabled())
        self.assertIn("database is locked", page.message.text())
        self.assertEqual(len(page.segments), 1)  # Not added, so still there to cut again.

    def test_stopping_waits_for_cutting_and_adds_what_was_cut(self):
        splitter, page = self.splitter_with_a_confirmed_match()
        page.create_clips()  # A real background thread.
        worker = page.worker
        page.stop_clips()
        self.assertIsNone(page.worker)
        self.assertFalse(worker.isRunning())
        cut = list((self.root / "matches").glob("*.mp4"))
        rows = self.db.connection.execute("SELECT COUNT(*) FROM videos WHERE project_id=?", (self.project,)).fetchone()[0]
        self.assertEqual(rows, len(cut))  # No cut file is left without its project row.

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

    def test_corner_drag_ends_when_released_over_the_black_bars(self):
        import numpy as np
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from control_vision.app.pages.calibrate import CornerCanvas
        canvas = CornerCanvas()
        canvas.resize(800, 300)  # A 4:3 frame here leaves black bars left and right.
        canvas.set_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        canvas.corners = [[160.0, 120.0]]
        canvas._press(160, 120)
        self.assertEqual(canvas._dragging, 0)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(5, 150))
        self.assertIsNone(canvas._dragging)

    def test_a_project_without_videos_shows_nothing_from_the_last_one(self):
        from control_vision.app.pages.calibrate import CalibratePage
        self.add_video("Q1.mp4")
        page = CalibratePage()
        page.set_project(self.db, self.project)
        page.canvas.corners = [list(point) for point in CORNERS]
        page._corners_changed()
        page.add_zone("Hive", "scoring", [[5, 5], [7, 5], [7, 7], [5, 7]])
        self.assertTrue(page.save())
        empty = self.db.add_project("Empty")
        page.set_project(self.db, empty)
        self.assertIsNone(page.video)
        self.assertEqual((page.canvas.corners, page.zones, page.map.zones), ([], {}, {}))
        self.assertIsNone(page.map.background)

    def test_calibration_saved_elsewhere_is_shown_unless_edits_are_unsaved(self):
        from control_vision.app.pages.calibrate import CalibratePage
        video, _ = self.add_video("Q1.mp4")
        page = CalibratePage()
        page.set_project(self.db, self.project)
        self.assertEqual(page.canvas.corners, [])
        self.db.save_calibration(video, CORNERS, {"Hive": {"type": "scoring", "points": [[5, 5], [7, 5], [7, 7]]}})
        page.refresh_videos()  # e.g. the page is shown again after Set up and process saved corners
        self.assertEqual(page.canvas.corners, CORNERS)
        self.assertIn("Hive", page.zones)
        self.assertFalse(page.dirty)
        page.canvas.corners[0] = [1.0, 1.0]
        page._corners_changed()
        page.refresh_videos()
        self.assertEqual(page.canvas.corners[0], [1.0, 1.0])  # Unsaved edits are kept.

    def test_keyboard_slider_moves_change_the_frame(self):
        from control_vision.app.pages import calibrate, identity
        self.add_video("Q1.mp4")
        for module, page_class in ((calibrate, calibrate.CalibratePage), (identity, identity.IdentityPage)):
            page = page_class()
            page.set_project(self.db, self.project)
            with patch.object(module, "read_frame", wraps=module.read_frame) as read:
                page.frame_slider.setValue(page.frame_slider.value() + 5)  # as the arrow keys do
            read.assert_called_once()

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

    def test_later_corrections_are_saved_shown_near_their_time_and_deleted(self):
        from control_vision.app.pages.identity import IdentityPage
        video, _ = self.add_video()
        ids = self.db.save_identity(video, [
            {"team_number": 7, "alliance": "red", "seed_x": 5, "seed_y": 5},
            {"team_number": 8, "alliance": "blue", "absent": True}])
        page = IdentityPage()
        page.set_project(self.db, self.project)
        self.assertEqual(page.frame_slider.maximum(), 30)  # start mode: the 3 s seed window
        self.assertFalse(page.corrections_box.isVisibleTo(page))
        page.later_mode.click()
        self.assertTrue(page.corrections_box.isVisibleTo(page))
        self.assertEqual(page.frame_slider.maximum(), 20)  # the whole 2 s video
        page.select(0)
        page.frame_slider.setValue(15)
        page._frame_clicked(30, 40)
        rows = self.db.reseeds_for_video(video)
        self.assertEqual([(r["robot_id"], r["t_s"], r["x_px"], r["y_px"]) for r in rows], [(ids[0], 1.5, 30.0, 40.0)])
        self.assertEqual(page.correction_list.count(), 1)
        self.assertIn("1.5 s — Robot 1 · 7", page.correction_list.item(0).text())
        self.assertEqual(page.canvas.corrections, [(30.0, 40.0, 0)])
        self.assertEqual(page.entries[0].seed, (5, 5))  # the start click is untouched
        page.frame_slider.setValue(0)
        self.assertEqual(page.canvas.corrections, [])  # 1.5 s away: not drawn
        # A robot saved as not visible has no start click, so it cannot be corrected.
        page.select(1)
        self.assertFalse(page.add_correction(10, 10))
        self.assertIn("start click", page.status.text())
        self.assertEqual(len(self.db.reseeds_for_video(video)), 1)
        # Leaving and coming back reloads them from the project.
        page.start_mode.click()
        self.assertIsNone(page.canvas.corrections)
        page.load_video(video)
        self.assertEqual(page.correction_list.count(), 1)
        page.later_mode.click()
        self.assertFalse(page.delete_correction_btn.isEnabled())
        page.correction_list.setCurrentRow(0)
        self.assertTrue(page.delete_correction_btn.isEnabled())
        page.delete_correction()
        self.assertEqual(self.db.reseeds_for_video(video), [])
        self.assertEqual(page.correction_list.count(), 0)

    def test_start_clicks_do_not_make_corrections(self):
        from control_vision.app.pages.identity import IdentityPage
        video, _ = self.add_video()
        self.db.save_identity(video, [{"team_number": 7, "alliance": "red", "seed_x": 5, "seed_y": 5}])
        page = IdentityPage()
        page.set_project(self.db, self.project)
        page.select(0)
        page._frame_clicked(30, 40)
        self.assertEqual(page.entries[0].seed, (30, 40))
        self.assertEqual(self.db.reseeds_for_video(video), [])

    def test_unsaved_start_changes_are_resolved_before_corrections(self):
        from control_vision.app.pages import identity
        video, _ = self.add_video()
        self.db.save_identity(video, [{"team_number": 7, "alliance": "red", "seed_x": 5, "seed_y": 5}])
        page = identity.IdentityPage()
        page.set_project(self.db, self.project)
        page.select(0)
        page._frame_clicked(30, 40)
        with patch.object(identity.QMessageBox, "question", return_value=identity.QMessageBox.StandardButton.Cancel):
            page.later_mode.click()
        self.assertTrue(page.start_mode.isChecked())
        with patch.object(identity.QMessageBox, "question", return_value=identity.QMessageBox.StandardButton.Discard):
            page.later_mode.click()
        self.assertTrue(page.later_mode.isChecked())
        self.assertEqual(page.entries[0].seed, (5, 5))
        self.assertFalse(page.dirty)

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

    def test_processing_after_the_page_loaded_still_asks(self):
        from control_vision.app.pages import identity
        video, _ = self.add_video()
        self.db.save_identity(video, [{"team_number": 7, "alliance": "red", "seed_x": 5, "seed_y": 5}])
        page = identity.IdentityPage()
        page.set_project(self.db, self.project)  # Loaded while still unprocessed.
        self.db.set_analysis_status(video, "ready")  # Then processed elsewhere.
        page.refresh_videos()
        self.assertEqual(page.video["analysis_status"], "ready")
        page.cards[0].blue.setChecked(True)
        page._card_changed()
        with patch.object(identity.QMessageBox, "question", return_value=identity.QMessageBox.StandardButton.Save) as asked:
            self.assertTrue(page.save())
        asked.assert_called_once()
        status = self.db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (video,)).fetchone()[0]
        self.assertEqual(status, "unprocessed")

    def test_a_database_error_on_save_is_reported(self):
        import sqlite3
        from control_vision.app.pages.identity import IdentityPage
        video, _ = self.add_video()
        page = IdentityPage()
        page.set_project(self.db, self.project)
        page.select(0)
        page._frame_clicked(40, 60)
        page.cards[0].team.setText("7"); page.cards[0].red.setChecked(True)
        page._card_changed()
        with patch.object(self.db, "save_identity", side_effect=sqlite3.OperationalError("database is locked")):
            self.assertFalse(page.save())
        self.assertIn("database is locked", page.status.text())
        self.assertTrue(page.dirty)

    def test_a_project_without_videos_shows_no_robots(self):
        from control_vision.app.pages.identity import IdentityPage
        video, _ = self.add_video()
        self.db.save_identity(video, [{"team_number": 7, "alliance": "red", "seed_x": 5, "seed_y": 5}])
        page = IdentityPage()
        page.set_project(self.db, self.project)
        self.assertEqual(page.entries[0].team, "7")
        page.set_project(self.db, self.db.add_project("Empty"))
        self.assertIsNone(page.video)
        self.assertTrue(all(entry.empty for entry in page.entries + page.saved_entries))
        self.assertEqual(page.cards[0].team.text(), "")

    def test_robots_saved_elsewhere_are_shown(self):
        from control_vision.app.pages.identity import IdentityPage
        video, _ = self.add_video()
        page = IdentityPage()
        page.set_project(self.db, self.project)
        self.db.save_identity(video, [{"team_number": 7, "alliance": "red", "seed_x": 5, "seed_y": 5}])
        page.refresh_videos()
        self.assertEqual((page.entries[0].team, page.entries[0].seed), ("7", (5, 5)))

    def test_typing_after_a_click_goes_to_the_next_robot(self):
        from control_vision.app.pages.identity import IdentityPage
        self.add_video()
        page = IdentityPage()
        page.set_project(self.db, self.project)
        page.show()
        self.addCleanup(page.close)
        page.cards[0].team.setText("21229"); page.cards[0].red.setChecked(True)
        page._card_changed()
        page.select(0)
        page._frame_clicked(40, 60)
        self.assertEqual(page.current, 1)
        self.assertIs(page.focusWidget(), page.cards[1].team)


class WindowSetupTests(SetupTest):
    def setUp(self):
        super().setUp()
        from control_vision.app import library
        from control_vision.app.main_window import MainWindow
        self.window = MainWindow()
        self.window.library = library.ProjectLibrary(self.root / "home")
        self.window.model_store = library.ModelStore(self.root / "home")

    def tearDown(self):
        self.window.calibrate_page.dirty = self.window.identity_page.dirty = False
        self.window.db = None
        self.window.close()

    def calibrate_with_unsaved_corners(self):
        window = self.window
        first, _ = self.add_video("Q1.mp4")
        second, _ = self.add_video("Q2.mp4")
        window.db, window.project_id = self.db, self.project
        window.refresh_home()
        page = window.calibrate_page
        page.load_video(first)
        page.canvas.corners = [list(point) for point in CORNERS]
        page._corners_changed()
        return page, second

    def test_cancel_keeps_the_user_on_their_page(self):
        from control_vision.app import main_window as mw
        from control_vision.app.pages import calibrate
        page, second = self.calibrate_with_unsaved_corners()
        self.window.tabs.setCurrentIndex(mw.IMPORT_TAB)
        with patch.object(calibrate.QMessageBox, "question", return_value=calibrate.QMessageBox.StandardButton.Cancel):
            self.window.open_in_calibrate(second)
        self.assertEqual(self.window.tabs.currentIndex(), mw.IMPORT_TAB)
        self.assertTrue(page.dirty)
        with patch.object(calibrate.QMessageBox, "question", return_value=calibrate.QMessageBox.StandardButton.Discard):
            self.window.open_in_calibrate(second)
        self.assertEqual(self.window.tabs.currentIndex(), mw.CALIBRATE_TAB)
        self.assertEqual(int(page.video["id"]), second)

    def test_saving_during_live_scoring_stays_quiet(self):
        from control_vision.app import main_window as mw
        window = self.window
        video, _ = self.add_video()
        window.db, window.project_id = self.db, self.project
        window.refresh_home()
        window.load_saved_video_id(video)
        self.assertTrue(mw._video_switch_blocked_quietly(SimpleNamespace(_live_active=True)))
        window._live_active = True
        self.addCleanup(setattr, window, "_live_active", False)
        with patch.object(mw.QMessageBox, "information") as dialog, \
                patch.object(window, "load_saved_video_id") as reload:
            window._setup_saved(video)
        dialog.assert_not_called()
        reload.assert_not_called()

    def test_continue_does_not_show_another_match_while_a_run_is_active(self):
        from control_vision.app import main_window as mw
        window = self.window
        first, _ = self.add_video("Q1.mp4")
        second, _ = self.add_video("Q2.mp4")
        window.db, window.project_id = self.db, self.project
        window.refresh_home()
        window.load_saved_video_id(first)
        window.tabs.setCurrentIndex(mw.IDENTITY_TAB)
        window._processing_active = True
        self.addCleanup(setattr, window, "_processing_active", False)
        with patch.object(mw.QMessageBox, "information") as dialog:
            window._continue_to_processing(second)
        dialog.assert_called_once()  # The user is told why.
        self.assertEqual(int(window.video_id), first)
        self.assertEqual(window.tabs.currentIndex(), mw.IDENTITY_TAB)

    def test_damaged_calibration_does_not_block_opening(self):
        window = self.window
        video, _ = self.add_video()
        self.db.save_calibration(video, CORNERS, {})
        self.db.connection.execute("UPDATE calibrations SET corners_json='{bad' WHERE video_id=?", (video,))
        window.db, window.project_id = self.db, self.project
        with patch.object(window, "_show_status") as status:
            window.refresh_home()
        self.assertIn("could not read", status.call_args.args[0])
        self.assertIsNone(window.calibrate_page.db)

    def test_closing_stops_clip_cutting_before_the_database_closes(self):
        from PySide6.QtGui import QCloseEvent
        window = self.window
        calls = []
        window.db = SimpleNamespace(close=lambda: calls.append("db_close"))
        with patch.object(window.splitter_page, "stop_clips", side_effect=lambda: calls.append("stop_clips")):
            window.closeEvent(QCloseEvent())
        self.assertEqual(calls, ["stop_clips", "db_close"])

    def test_closing_the_app_asks_about_unsaved_setup_edits(self):
        from PySide6.QtGui import QCloseEvent
        from control_vision.app.pages import calibrate
        self.calibrate_with_unsaved_corners()
        event = QCloseEvent()
        with patch.object(calibrate.QMessageBox, "question", return_value=calibrate.QMessageBox.StandardButton.Cancel):
            self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertIsNotNone(self.window.db)

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
