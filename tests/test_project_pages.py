"""Home dashboard, Projects library, New project wizard, and how the window wires them."""
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from control_vision.storage.database import ProjectDatabase

META = {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720}


def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class PageTest(unittest.TestCase):
    def setUp(self):
        self.app = qt_app()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def make_project(self, name, folder, ready=False, flagged=0, team=21229, archived=False):
        path = self.root / folder / "control_vision.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        db = ProjectDatabase(path)
        try:
            project = db.add_project(name)
            db.set_project_details(project, event_name=f"{name} event", archived=int(archived))
            video = db.add_video(project, self.root / "missing.mp4", META, None, None)
            robot = db.add_robot(video, team, "red", 1, 1)
            if ready:
                db.set_analysis_status(video, "ready")
                for _ in range(flagged):
                    db.add_scoring_event(robot, 40.0)
        finally:
            db.close()
        return path


class HomePageTests(PageTest):
    def test_greeting_and_stats_are_counted_not_invented(self):
        from control_vision.app import library
        from control_vision.app.pages.home import HomePage, greeting_line, model_stat
        summaries = [library.summarize_project(self.make_project("A", "a", ready=True, flagged=1)),
                     library.summarize_project(self.make_project("B", "b"))]
        stats = library.home_stats(summaries)
        self.assertEqual(greeting_line(datetime(2026, 11, 15), stats),
                         "Sunday, Nov 15 — 2 projects, 1 ready to review")
        self.assertEqual(model_stat(None)[0], "—")
        unmeasured = library.ModelVersion(id="v1", name="v1", path="v1.onnx")
        self.assertIn("not measured", model_stat(unmeasured)[1])
        page = HomePage()
        page.refresh(summaries, stats, None, None)
        self.assertEqual(page.stat_matches.value.text(), "1")
        self.assertFalse(page.resume_btn.isEnabled())
        self.assertEqual(page.resume_btn.toolTip(), "No recent project")
        self.assertEqual(len(page.visible_summaries()), 2)
        page.status_filter.setCurrentIndex(page.status_filter.findData("Ready to review"))
        self.assertEqual([s.name for s in page.visible_summaries()], ["A"])

    def test_empty_state_offers_one_action(self):
        from control_vision.app import library
        from control_vision.app.pages.home import HomePage
        page = HomePage()
        page.refresh([], library.HomeStats(), None, None)
        self.assertFalse(page.empty.isHidden())
        self.assertTrue(page.scroll.isHidden())
        clicks = []
        page.new_project.connect(lambda: clicks.append(True))
        page.empty.action.click()
        self.assertEqual(clicks, [True])


class ProjectsPageTests(PageTest):
    def summaries(self):
        from control_vision.app import library
        return [library.summarize_project(self.make_project("Salk Meet", "a", ready=True, flagged=1)),
                library.summarize_project(self.make_project("Bellevue", "b", team=16655)),
                library.summarize_project(self.make_project("Old", "c", archived=True)),
                library.summarize_project(self.root / "gone" / "control_vision.sqlite")]

    def test_search_and_filters_combine(self):
        from control_vision.app.pages.projects import ProjectsPage
        page = ProjectsPage()
        page.refresh(self.summaries(), None)
        self.assertEqual(len(page.visible()), 3)  # Archived is hidden until asked for.
        page.search.setText("16655")
        self.assertEqual([s.name for s in page.visible()], ["Bellevue"])
        page.status.setCurrentIndex(page.status.findData("Ready to review"))
        self.assertEqual(page.visible(), [])
        page.clear_filters()
        page.show_archived.setChecked(True)
        self.assertEqual(len(page.visible()), 4)

    def test_remove_from_list_forgets_a_missing_project(self):
        from control_vision.app.pages.projects import ProjectsPage
        page = ProjectsPage()
        page.refresh(self.summaries(), None)
        forgotten, deleted = [], []
        page.forget.connect(forgotten.append)
        page.delete.connect(deleted.append)
        missing_row = next(row for row in range(page.table.rowCount())
                           if "Missing" in page.table.item(row, 6).text())
        page.table.selectRow(missing_row)
        self.assertEqual(page.delete_btn.text(), "Remove from list")
        page.delete_btn.click()
        self.assertEqual(len(forgotten), 1)
        self.assertEqual(deleted, [])

    def test_unreadable_files_match_no_status_filter(self):
        from control_vision.app import library
        from control_vision.app.pages.projects import matches_filter
        junk = self.root / "junk.sqlite"
        junk.write_bytes(b"not a database")
        summary = library.summarize_project(junk)
        self.assertTrue(matches_filter(summary, "", None, None, "", None, False))
        self.assertFalse(matches_filter(summary, "", None, None, "", "Not processed", False))

    def test_clear_filters_reveals_a_fully_archived_library(self):
        from control_vision.app import library
        from control_vision.app.pages.projects import ProjectsPage
        page = ProjectsPage()
        page.refresh([library.summarize_project(self.make_project("Old", "c", archived=True))], None)
        self.assertEqual(page.visible(), [])
        page.clear_filters()
        self.assertEqual(len(page.visible()), 1)

    def test_unreadable_files_are_not_shown_as_new_projects(self):
        from control_vision.app import library
        from control_vision.app.pages.home import ProjectCard
        junk = self.root / "junk" / "control_vision.sqlite"
        junk.parent.mkdir()
        junk.write_bytes(b"not a database")
        summary = library.summarize_project(junk)
        card = ProjectCard(summary)
        from PySide6.QtWidgets import QLabel
        texts = [label.text() for label in card.findChildren(QLabel)]
        self.assertIn("✕ Cannot read", texts)
        self.assertNotIn("○ Not processed", texts)

    def test_model_column_names_the_active_version(self):
        from control_vision.app import library
        from control_vision.app.pages.projects import model_cell
        summary = library.ProjectSummary(path=Path("x"), model_name="v0.3.0")
        self.assertEqual(model_cell(summary, "v0.4.0"), "v0.3.0, active is v0.4.0")
        self.assertEqual(model_cell(summary, "v0.3.0"), "v0.3.0")
        self.assertEqual(model_cell(library.ProjectSummary(path=Path("x")), "v0.4.0"), "—")


class WizardTests(PageTest):
    def wizard(self, models=True, names=()):
        from control_vision.app import library
        from control_vision.app.pages.wizard import NewProjectWizard, WizardVideo
        store = library.ModelStore(self.root / "home")
        if models:
            weights = self.root / "v1.onnx"
            weights.write_bytes(b"onnx")
            (self.root / "v1.metrics.json").write_text(json.dumps({"map50": 0.8}), encoding="utf-8")
            store.add_file(weights)
        wizard = NewProjectWizard(store, lambda: set(names), lambda name: self.root / "projects" / name)
        (self.root / "Q1.mp4").write_bytes(b"video")
        self.video = WizardVideo(path=self.root / "Q1.mp4", duration_s=150.0, size=1000, metadata=dict(META))
        return wizard

    def fill_first_step(self, wizard, name="Meet"):
        wizard.name.setText(name)
        wizard.event_name.setText("Salk League")
        wizard.event_code.setText("wa salk")

    def test_next_stays_disabled_with_a_reason_and_back_keeps_data(self):
        wizard = self.wizard(names={"Meet"})
        self.assertFalse(wizard.next_btn.isEnabled())
        self.assertEqual(wizard.reason.text(), "Enter a project name")
        self.fill_first_step(wizard)
        self.assertIn('"Meet (2)"', wizard.reason.text())
        wizard.use_suggestion.click()
        self.assertEqual(wizard.name.text(), "Meet (2)")
        self.assertTrue(wizard.next_btn.isEnabled())
        wizard.next()
        self.assertEqual(wizard.reason.text(), "Add at least 1 video")
        wizard.back()
        self.assertEqual(wizard.event_name.text(), "Salk League")

    def test_youtube_links_are_checked_but_never_downloaded(self):
        wizard = self.wizard()
        wizard.youtube.setText("https://www.youtube.com/watch?v=abcdefghijk")
        wizard._check_link()
        self.assertIn("not available in this build", wizard.youtube_status.text())
        wizard.youtube.setText("not a link")
        wizard._check_link()
        self.assertEqual(wizard.youtube_status.property("tone"), "danger")
        self.assertEqual(wizard.videos, [])

    def test_no_models_blocks_the_model_step(self):
        from control_vision.app.pages.wizard import step_problem
        wizard = self.wizard(models=False)
        self.fill_first_step(wizard)
        wizard.next()
        wizard.videos = [self.video]
        wizard.next()
        self.assertEqual(wizard.step, 2)
        self.assertEqual(wizard.reason.text(), "No models yet — train one first")
        self.assertFalse(wizard.no_models.isHidden())
        self.assertIsNone(step_problem(1, wizard.spec(), set(), []))

    def test_create_writes_the_project_and_nothing_before_it(self):
        from control_vision.app import library
        wizard = self.wizard()
        self.fill_first_step(wizard)
        wizard.next()
        wizard.videos = [self.video]
        wizard._refresh_videos()
        wizard.next()
        wizard.model_list.setCurrentRow(0)
        wizard.next()
        folder = self.root / "projects" / "Meet"
        self.assertFalse(folder.exists())
        self.assertTrue(wizard.create_btn.isEnabled())
        with patch.object(wizard, "created") as created:
            wizard.create()
        path = created.emit.call_args.args[0]
        summary = library.summarize_project(path)
        self.assertEqual((summary.name, summary.event_code, summary.matches, summary.model_name),
                         ("Meet", "WASALK", 1, "v1"))
        settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        self.assertTrue(settings["model_path"].endswith("v1.onnx"))
        self.assertEqual(wizard.step, 0)  # The wizard starts over after Create.
        self.assertIsNone(wizard.model_list.currentItem())  # ...and keeps no model pick.

    def reach_confirm(self, wizard, name="Meet"):
        self.fill_first_step(wizard, name)
        wizard.next()
        wizard.videos = [self.video]
        wizard._refresh_videos()
        wizard.next()
        wizard.model_list.setCurrentRow(0)
        wizard.next()

    def test_a_failed_create_leaves_the_folder_reusable(self):
        from control_vision.app.pages import wizard as wizard_module
        wizard = self.wizard()
        self.reach_confirm(wizard)
        folder = self.root / "projects" / "Meet"
        with patch("control_vision.storage.database.ProjectDatabase.set_video_details", side_effect=OSError("disk full")):
            wizard.create()
        self.assertFalse(wizard.error.isHidden())
        self.assertFalse(folder.exists())
        self.assertIsNone(wizard_module.step_problem(0, wizard.spec(), set(), []))

    def test_cancel_forgets_the_model_and_videos_can_be_removed(self):
        wizard = self.wizard()
        self.reach_confirm(wizard)
        wizard.back(); wizard.back()
        self.assertFalse(wizard.remove_video_btn.isEnabled())
        wizard.video_list.setCurrentRow(0)
        self.assertTrue(wizard.remove_video_btn.isEnabled())
        wizard.remove_video_btn.click()
        self.assertEqual(wizard.videos, [])
        wizard.cancel()
        self.assertIsNone(wizard.model_list.currentItem())

    def test_cancel_hides_the_old_name_suggestion(self):
        wizard = self.wizard(names={"Meet"})
        wizard.name.setText("Meet")
        self.assertFalse(wizard.use_suggestion.isHidden())
        wizard.cancel()
        self.assertTrue(wizard.use_suggestion.isHidden())

    def test_moved_videos_are_skipped_and_names_are_escaped(self):
        wizard = self.wizard()
        self.reach_confirm(wizard, name="Meet <Final> & Co")
        self.assertIn("&lt;Final&gt; &amp; Co", wizard.summary.text())
        (self.root / "Q1.mp4").unlink()
        wizard._show_step()
        self.assertIn("File moved or deleted", wizard.video_list.item(0).text())
        self.assertFalse(wizard.create_btn.isEnabled())
        self.assertEqual(wizard.reason.text(), "Add at least 1 video")


class WindowProjectTests(PageTest):
    def setUp(self):
        super().setUp()
        from control_vision.app import library
        from control_vision.app.main_window import MainWindow
        self.window = MainWindow()
        self.window.library = library.ProjectLibrary(self.root / "home")
        self.window.model_store = library.ModelStore(self.root / "home")

    def tearDown(self):
        if self.window.db:
            self.window.db.close()
            self.window.db = None
        self.window.close()

    def test_window_starts_on_home_and_settings_opens_as_a_window(self):
        from control_vision.app import main_window as mw
        window = self.window
        self.assertEqual(window.tabs.currentIndex(), mw.HOME_TAB)
        self.assertEqual(window.page_title.text(), "Home")
        self.assertNotIn(mw.SETTINGS_TAB, window._nav_rows)
        window._go_to_tab(mw.SETTINGS_TAB)
        self.assertTrue(window.settings_dialog.isVisible())
        self.assertIs(window.theme_combo.window(), window.settings_dialog)
        window.settings_dialog.reject()
        self.assertEqual(window.top_scout.text(), "Local only")

    def test_question_mark_types_in_fields_and_opens_help_elsewhere(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        window = self.window
        window.show()
        with patch.object(window, "show_shortcuts") as shown:
            window.top_search.setFocus()
            QTest.keyClicks(window.top_search, "?")
            self.assertEqual(window.top_search.text(), "?")
            shown.assert_not_called()
            QTest.keyClick(window, Qt.Key.Key_Question, Qt.KeyboardModifier.ShiftModifier)
            shown.assert_called_once()

    def test_opening_from_the_library_registers_and_updates_the_top_bar(self):
        window = self.window
        path = self.make_project("Salk Meet", "a")
        self.assertTrue(window.open_project_path(path))
        self.assertEqual(window.library.last(), path.resolve())
        self.assertEqual(window.top_project.text(), "Salk Meet")
        self.assertEqual(window.top_event.text(), "Salk Meet event")
        self.assertEqual(window.db_label.text(), "control_vision.sqlite")
        self.assertIsNotNone(window.db.project_details(window.project_id)["last_opened"])

    def test_a_missing_project_file_is_not_created(self):
        window = self.window
        missing = self.root / "gone" / "control_vision.sqlite"
        self.assertFalse(window.open_project_path(missing))
        self.assertFalse(missing.exists())
        self.assertEqual(window.progress_label.property("tone"), "danger")

    def test_archive_and_delete_ask_and_keep_the_open_project(self):
        from control_vision.app import library
        from control_vision.app import main_window as mw
        window = self.window
        other = self.make_project("Other", "b")
        window.library.register(other)
        window.set_project_archived(other, True)
        self.assertTrue(library.summarize_project(other).archived)
        with patch.object(mw.QMessageBox, "question", return_value=mw.QMessageBox.StandardButton.No), \
                patch.object(mw.project_library, "remove_file_to_trash") as trash:
            window.delete_project_file(other)
        trash.assert_not_called()
        with patch.object(mw.QMessageBox, "question", return_value=mw.QMessageBox.StandardButton.Yes) as asked, \
                patch.object(mw.project_library, "remove_file_to_trash", return_value=True):
            window.delete_project_file(other)
        self.assertIn("1 match", asked.call_args.args[2])
        self.assertEqual(window.library.paths(), [])
        opened = self.make_project("Open", "c")
        window.open_project_path(opened)
        with patch.object(mw.QMessageBox, "information") as told, patch.object(mw.QMessageBox, "question") as asked:
            window.delete_project_file(opened)
        told.assert_called_once()
        asked.assert_not_called()

    def test_deleting_a_missing_project_just_removes_it_from_the_list(self):
        from control_vision.app import main_window as mw
        window = self.window
        missing = self.root / "gone" / "control_vision.sqlite"
        window.library.register(missing)
        with patch.object(mw.QMessageBox, "question") as asked, patch.object(mw.QMessageBox, "critical") as failed:
            window.delete_project_file(missing)
        asked.assert_not_called()
        failed.assert_not_called()
        self.assertEqual(window.library.paths(), [])


if __name__ == "__main__":
    unittest.main()
