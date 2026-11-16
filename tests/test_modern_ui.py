"""The themed shell: sidebar navigation, workflow steps and the pending-review look."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from control_vision.storage.database import ProjectDatabase

META = {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720}


class ThemeTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        from control_vision.app import theme
        cls.theme = theme

    def test_pending_tone_is_distinct_from_the_accent_in_both_themes(self):
        for tokens in (self.theme.LIGHT, self.theme.DARK):
            colours = {tokens.accent.lower(), tokens.accent_hover.lower(), tokens.selection.lower()}
            self.assertNotIn(tokens.pending_bg.lower(), colours)
            self.assertNotIn(tokens.pending_fg.lower(), colours)
            self.assertNotEqual(tokens.pending_fg.lower(), tokens.accent_text.lower())

    def test_rows_needing_review_use_the_pending_tone(self):
        tokens = self.theme.LIGHT
        pending = (tokens.pending_bg, tokens.pending_fg)
        self.assertEqual(self.theme.row_colours(tokens, "flagged", "high"), pending)
        self.assertEqual(self.theme.row_colours(tokens, "confirmed", "low"), pending)
        self.assertEqual(self.theme.row_colours(tokens, "confirmed", "high"), (None, None))
        self.assertEqual(self.theme.row_colours(tokens, "rejected", "low"), (None, tokens.text_muted))

    def test_status_and_confidence_labels(self):
        self.assertIn("Pending review", self.theme.status_label("flagged"))
        self.assertIn("Confirmed", self.theme.status_label("confirmed"))
        self.assertIn("Rejected", self.theme.status_label("rejected"))
        self.assertEqual(self.theme.status_label("something-new"), "something-new")
        self.assertIn("low", self.theme.confidence_label("low"))
        self.assertNotEqual(self.theme.confidence_label("low"), "low")
        self.assertEqual(self.theme.confidence_label("high"), "high")

    def test_modes_resolve_and_the_stylesheet_uses_the_tokens(self):
        self.assertIs(self.theme.resolve_tokens("light"), self.theme.LIGHT)
        self.assertIs(self.theme.resolve_tokens("dark"), self.theme.DARK)
        self.assertIn(self.theme.resolve_tokens("system"), (self.theme.LIGHT, self.theme.DARK))
        sheet = self.theme.build_stylesheet(self.theme.DARK)
        self.assertIn(self.theme.DARK.accent, sheet)
        self.assertIn(self.theme.DARK.pending_bg, sheet)
        with self.assertRaises(ValueError):
            self.theme.save_mode("neon")

    def test_tones_are_validated(self):
        from PySide6.QtWidgets import QLabel
        label = QLabel()
        self.theme.set_tone(label, "pending")
        self.assertEqual(label.property("tone"), "pending")
        self.theme.set_tone(label, None)
        self.assertIsNone(label.property("tone"))
        with self.assertRaises(ValueError):
            self.theme.set_tone(label, "loud")


class WorkflowStepTests(unittest.TestCase):
    def steps(self, **overrides):
        from control_vision.app.main_window import workflow_steps
        values = dict(project_open=True, video_name="Q1.mp4", corners=4, field_zones=2, rois=1,
                      robots=4, analysis_status="ready", processing=False)
        values.update(overrides)
        return {step["key"]: step for step in workflow_steps(**values)}

    def test_order_follows_the_workflow(self):
        from control_vision.app.main_window import workflow_steps
        keys = [step["key"] for step in workflow_steps(project_open=False, video_name=None, corners=0, field_zones=0,
                                                       rois=0, robots=0, analysis_status=None, processing=False)]
        self.assertEqual(keys, ["video", "calibrate", "zones", "robots", "process"])

    def test_without_a_project_only_the_video_step_is_current(self):
        steps = self.steps(project_open=False, video_name=None, corners=0, field_zones=0, rois=0, robots=0,
                           analysis_status=None)
        self.assertEqual(steps["video"]["state"], "current")
        self.assertIn("project", steps["video"]["detail"].lower())
        self.assertEqual({steps[key]["state"] for key in ("calibrate", "robots", "process")}, {"todo"})

    def test_first_missing_required_step_is_current_and_zones_are_optional(self):
        steps = self.steps(corners=2, field_zones=0, rois=0, robots=0, analysis_status="unprocessed")
        self.assertEqual(steps["video"]["state"], "done")
        self.assertEqual(steps["calibrate"]["state"], "current")
        self.assertIn("2/4", steps["calibrate"]["detail"])
        self.assertEqual(steps["zones"]["state"], "optional")
        self.assertEqual(steps["robots"]["state"], "todo")

    def test_processing_failure_and_ready_states(self):
        self.assertEqual(self.steps(processing=True, analysis_status="running")["process"]["state"], "running")
        self.assertEqual(self.steps(analysis_status="failed")["process"]["state"], "failed")
        # A run interrupted before it could record "failed" is not shown as ready.
        self.assertEqual(self.steps(analysis_status="running")["process"]["state"], "failed")
        ready = self.steps()
        self.assertEqual({step["state"] for step in ready.values()}, {"done"})


class ShellWindowTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])
        from control_vision.app.main_window import MainWindow
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = ProjectDatabase(Path(folder.name) / "project.sqlite")
        self.project = self.db.add_project("event")
        self.video = self.db.add_video(self.project, Path("Q1.mp4"), META, "EV", "Q1")
        self.robot = self.db.add_robot(self.video, 1001, "red", 10, 10)
        self.window = MainWindow()

    def tearDown(self):
        self.window.db = None
        self.window.close()
        self.db.close()

    def attach(self):
        self.window.db, self.window.project_id, self.window.video_id = self.db, self.project, self.video

    def test_sidebar_replaces_the_tab_bar_and_mirrors_disabled_tabs(self):
        from PySide6.QtCore import Qt
        window = self.window
        self.assertTrue(window.tabs.tabBar().isHidden())
        self.assertEqual(window.tabs.tabText(2), "Cycles && Stats")
        enabled = Qt.ItemFlag.ItemIsEnabled
        window._set_results_enabled(False)
        self.assertFalse(window.nav.item(window._nav_rows[3]).flags() & enabled)
        self.assertTrue(window.nav.item(window._nav_rows[0]).flags() & enabled)
        window._set_results_enabled(True)
        self.assertTrue(window.nav.item(window._nav_rows[3]).flags() & enabled)
        window.nav.setCurrentRow(window._nav_rows[6])
        self.assertEqual(window.tabs.currentIndex(), 6)
        self.assertEqual(window.page_title.text(), "Score and export")
        window.tabs.setCurrentIndex(0)
        self.assertEqual(window.nav.currentRow(), window._nav_rows[0])

    def test_results_screens_are_unavailable_before_any_video_is_processed(self):
        window = self.window
        for index in range(2, 7):
            self.assertFalse(window.tabs.isTabEnabled(index), index)
        window._go_to_tab(4)
        self.assertNotEqual(window.tabs.currentIndex(), 4)

    def test_sidebar_tooltips_follow_availability(self):
        window = self.window
        self.assertIn("process the video", window.nav.item(window._nav_rows[4]).toolTip())
        window._set_results_enabled(True)
        self.assertEqual(window.nav.item(window._nav_rows[4]).toolTip(), "Scoring timeline (Ctrl+4)")
        window._set_review_badge(2)
        window._sync_nav()
        self.assertIn("2 event(s) pending", window.nav.item(window._nav_rows[3]).toolTip())
        window._set_review_badge(0)
        self.assertEqual(window.nav.item(window._nav_rows[3]).toolTip(), "Review queue (Ctrl+3)")

    def test_minimum_window_width_fits_every_panel(self):
        # Fixed and minimum widths only, so the check does not depend on installed fonts.
        from PySide6.QtWidgets import QFrame, QScrollArea, QSplitter
        window = self.window
        sidebar = window.findChild(QFrame, "Sidebar")
        steps = next(area for area in window.findChildren(QScrollArea) if area.maximumWidth() == area.minimumWidth())
        splitter = next(s for s in window.findChildren(QSplitter) if s.indexOf(window.canvas) >= 0)
        needed = (sidebar.minimumWidth() + 2 * 24 + steps.minimumWidth() + 16 + window.canvas.minimumWidth()
                  + splitter.handleWidth() + window.field_map.minimumWidth())
        self.assertLessEqual(needed, window.minimumWidth())
        self.assertLessEqual(window.minimumWidth(), window.width())

    def test_navigation_to_a_disabled_tab_is_refused(self):
        window = self.window
        window.tabs.setCurrentIndex(1)
        window._set_results_enabled(False)
        window._go_to_tab(3)
        self.assertEqual(window.tabs.currentIndex(), 1)
        window.nav.setCurrentRow(window._nav_rows[3])
        self.assertEqual(window.tabs.currentIndex(), 1)
        self.assertEqual(window.nav.currentRow(), window._nav_rows[1])

    def test_flagged_events_are_counted_and_tinted_as_pending(self):
        from control_vision.app import theme
        event = self.db.add_scoring_event(self.robot, 40.0)
        self.db.set_analysis_status(self.video, "ready")
        self.attach()
        self.window._set_results_enabled(True)
        self.window.refresh_result_tables()
        self.assertEqual(self.window.review_table.rowCount(), 1)
        self.assertEqual(self.window.review_table.item(0, 0).data(0x0100), event)
        self.assertIn("(1)", self.window.nav.item(self.window._nav_rows[3]).text())
        status = self.window.review_table.item(0, 7)
        self.assertIn("Pending review", status.text())
        tokens = self.window._tokens
        self.assertEqual(status.background().color().name().lower(), tokens.pending_bg.lower())
        # Switching theme re-tints the rows without rebuilding them.
        with patch.object(theme, "save_mode") as saved:
            other = "dark" if tokens is theme.LIGHT else "light"
            self.window.theme_combo.setCurrentIndex(self.window.theme_combo.findData(other))
        saved.assert_called_once_with(other)
        self.assertEqual(status.background().color().name().lower(),
                         theme.resolve_tokens(other).pending_bg.lower())
        self.assertEqual(self.app.property("control_vision_theme"), other)

    def test_review_controls_keep_labels_mnemonics_and_accessible_names(self):
        from PySide6.QtWidgets import QLabel
        window = self.window
        buddies = {label.buddy(): label.text() for label in window.findChildren(QLabel) if label.buddy() is not None}
        for widget in (window.review_type, window.review_robot, window.review_points, window.review_alliance,
                       window.review_confidence, window.match_type_combo, window.ruling_combo, window.theme_combo):
            self.assertIn("&", buddies.get(widget, ""), widget.accessibleName())
            self.assertTrue(widget.accessibleName())
        self.assertEqual(window.match_type_combo.accessibleName(), "Match type")

    def test_workflow_panel_follows_setup_progress(self):
        window = self.window
        number, state = window._step_widgets["calibrate"]
        self.assertEqual(number.property("state"), "todo")
        self.attach()
        window.video_path = Path("Q1.mp4")
        window.corners = [[0, 0], [1, 0]]
        window.refresh_home()
        self.assertEqual(number.property("state"), "current")
        self.assertIn("2/4", state.text())
        self.assertIn("Q1.mp4", window.page_context.text())
        self.assertIn("event", window.context_label.text())

    def test_pending_referee_ruling_and_ocr_proposal_use_the_pending_tone(self):
        self.db.set_match_type(self.video, "playoff")
        self.db.set_match_ruling(self.video, "playoff", playoff_disqualified={"red", "blue"})
        self.db.set_analysis_status(self.video, "ready")
        self.attach()
        self.window._set_results_enabled(True)
        self.window.load_match_ruling()
        self.window.update_score_comparison()
        self.assertIn("PENDING REFEREE RULING", self.window.ruling_status.text())
        self.assertEqual(self.window.ruling_status.property("tone"), "pending")
        with patch.object(self.window, "current_scoreboard", side_effect=ValueError("invalid stored ruling")):
            self.window.update_score_comparison()
        self.assertEqual(self.window.ruling_status.text(), "invalid stored ruling")
        self.assertEqual(self.window.ruling_status.property("tone"), "danger")
        self.window.show_scoreboard_ocr({"frames_sampled": 5, "red": {"value": 40, "confidence": 0.9},
                                         "blue": {"value": 30, "confidence": 0.9}})
        self.assertEqual(self.window.scoreboard_ocr_status.property("tone"), "pending")
        self.assertTrue(self.window.use_ocr_btn.isEnabled())
        self.window.show_scoreboard_ocr(None)
        self.assertEqual(self.window.scoreboard_ocr_status.property("tone"), "info")
        self.assertFalse(self.window.use_ocr_btn.isEnabled())

    def test_setup_messages_after_a_failure_do_not_keep_the_failure_tone(self):
        window = self.window
        window.processing_failed("model failed")
        self.assertEqual(window.progress_label.property("tone"), "danger")
        self.attach()
        window.video_path = Path("Q1.mp4")
        window._mode = "corner"
        window.corners = [[0, 0], [100, 0], [100, 100]]
        with patch("control_vision.app.main_window.cv2.VideoCapture") as capture:
            capture.return_value.read.return_value = (False, None)
            window.canvas_click(0, 100)
        self.assertIn("Calibration saved", window.progress_label.text())
        self.assertEqual(window.progress_label.property("tone"), "info")
        window._mode = "corner"
        window.corners = [[0, 0], [1, 0], [2, 0]]
        with patch("control_vision.app.main_window.cv2.VideoCapture") as capture, \
                patch("control_vision.app.main_window.QMessageBox.warning"):
            capture.return_value.read.return_value = (False, None)
            window.canvas_click(3, 0)
        self.assertIn("Calibration not saved", window.progress_label.text())
        self.assertEqual(window.progress_label.property("tone"), "danger")

    def test_failed_project_switch_shows_no_project_or_video(self):
        from control_vision.app import main_window as mw
        window = self.window
        self.attach()
        window.video_path = Path("Q1.mp4")
        window.refresh_home()
        self.assertIn("event", window.page_context.text())
        window.db = ProjectDatabase(self.db.path)  # the window closes its own handle
        with patch.object(mw.QFileDialog, "getOpenFileName", return_value=("missing.sqlite", "")), \
                patch.object(mw, "ProjectDatabase", side_effect=OSError("disk unavailable")), \
                patch.object(mw.QMessageBox, "critical") as critical:
            window.open_project()
        critical.assert_called_once()
        self.assertIsNone(window.db)
        self.assertIsNone(window.project_id)
        self.assertIsNone(window.video_path)
        self.assertEqual(window.page_context.text(), "No project open")
        self.assertEqual(window.context_label.text(), "No project open")
        self.assertEqual(window._step_widgets["video"][0].property("state"), "current")
        self.assertIn("Could not open project", window.progress_label.text())
        self.assertEqual(window.progress_label.property("tone"), "danger")

    def test_loading_a_video_whose_analysis_did_not_finish_shows_an_error(self):
        from control_vision.app import main_window as mw
        import numpy as np
        window = self.window
        self.attach()
        Path(self.db.path.parent / "Q1.mp4").write_bytes(b"")
        self.db.connection.execute("UPDATE videos SET path=? WHERE id=?", (str(self.db.path.parent / "Q1.mp4"), self.video))
        self.db.connection.commit()
        frame = np.zeros((72, 128, 3), dtype=np.uint8)
        for status, tone in (("failed", "danger"), ("running", "danger"), ("unprocessed", "info")):
            self.db.set_analysis_status(self.video, status)
            with patch.object(mw, "probe_video", return_value=dict(META, warnings=[])), \
                    patch.object(mw.cv2, "VideoCapture") as capture:
                capture.return_value.read.return_value = (True, frame)
                window.load_saved_video_id(self.video)
            self.assertEqual(window.progress_label.property("tone"), tone, status)

    def test_progress_bar_shows_only_while_processing(self):
        from control_vision.app import main_window as mw
        window = self.window
        self.assertTrue(window.status_progress.isHidden())
        self.attach()
        window.video_path = Path("Q1.mp4")
        window.corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
        window.seed_points = [(self.robot, 10, 10)]
        window.tracking_mode.setCurrentIndex(window.tracking_mode.findData("seeded_optical_flow"))
        # Start a run without a real worker thread; process() itself must show the bar.
        with patch.object(mw, "ProcessingWorker"), patch.object(mw, "QThread") as thread:
            window.process()
        self.addCleanup(lambda: [handler.close() for handler in window.logger.handlers])
        thread.return_value.start.assert_called_once()
        self.assertFalse(window.status_progress.isHidden())
        self.assertEqual(window.status_progress.value(), 0)
        self.assertEqual(window._step_widgets["process"][0].property("state"), "running")
        window.show_progress("Tracking", 42.4, "frame 10")
        self.assertEqual(window.status_progress.value(), 42)
        self.assertIn("42%", window._step_widgets["process"][1].text())
        window.processing_failed("Processing was cancelled.")
        self.assertTrue(window.status_progress.isHidden())
        self.assertEqual(window.progress_label.property("tone"), "danger")


if __name__ == "__main__":
    unittest.main()
