"""Project library, project details and model versions behind the Home and Projects pages."""
import json
import tempfile
import unittest
from pathlib import Path

from control_vision.app import library
from control_vision.storage.database import ProjectDatabase, normalise_event_code

META = {"duration_s": 150.0, "fps": 30.0, "width": 1280, "height": 720}


class TempFolderTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def make_project(self, name="Meet 1", folder="p1", videos=1, ready=0, flagged=0):
        path = self.root / folder / "control_vision.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        db = ProjectDatabase(path)
        try:
            project = db.add_project(name)
            db.set_project_details(project, event_name="Salk League", event_code=" wa salk ", event_date="2026-11-15")
            for index in range(videos):
                video = db.add_video(project, self.root / f"Q{index}.mp4", META, "WASALK", f"Q{index}")
                robot = db.add_robot(video, 21229, "red", 10, 10)
                if index < ready:
                    db.set_analysis_status(video, "ready")
                    for _ in range(flagged):
                        db.add_scoring_event(robot, 40.0)
        finally:
            db.close()
        return path


class ProjectDetailTests(TempFolderTest):
    def test_event_codes_are_normalised_and_unknown_fields_refused(self):
        self.assertEqual(normalise_event_code(" wa salk "), "WASALK")
        path = self.make_project()
        db = ProjectDatabase(path)
        try:
            details = db.project_details(1)
            self.assertEqual(details["event_code"], "WASALK")
            self.assertEqual(details["season"], "BIOBUZZ 2026-27")
            self.assertEqual(details["archived"], 0)
            with self.assertRaises(ValueError):
                db.set_project_details(1, score=99)
            db.touch_project(1, "2026-11-15T10:00:00")
            self.assertEqual(db.project_details(1)["last_opened"], "2026-11-15T10:00:00")
        finally:
            db.close()

    def test_migration_is_idempotent(self):
        path = self.make_project()
        ProjectDatabase(path).close()
        db = ProjectDatabase(path)
        try:
            self.assertEqual(db.project_details(1)["name"], "Meet 1")
        finally:
            db.close()


class SummaryTests(TempFolderTest):
    def test_status_follows_processing_and_review(self):
        self.assertEqual(library.summarize_project(self.make_project(folder="a")).status, "Not processed")
        ready = library.summarize_project(self.make_project(folder="b", ready=1, flagged=2))
        self.assertEqual(ready.status, "Ready to review")
        self.assertEqual(ready.flagged, 2)
        self.assertEqual(ready.teams, {21229})
        reviewed = library.summarize_project(self.make_project(folder="c", ready=1))
        self.assertEqual(reviewed.status, "Reviewed")

    def test_missing_and_unreadable_files_are_reported_not_raised(self):
        missing = library.summarize_project(self.root / "gone.sqlite")
        self.assertTrue(missing.missing)
        self.assertFalse((self.root / "gone.sqlite").exists())  # Reading never creates a file.
        junk = self.root / "junk.sqlite"
        junk.write_bytes(b"not a database")
        self.assertIsNotNone(library.summarize_project(junk).unreadable)

    def test_home_stats_skip_archived_and_missing_projects(self):
        first = self.make_project(folder="a", videos=2, ready=2, flagged=1)
        archived = self.make_project(folder="b", ready=1)
        db = ProjectDatabase(archived)
        db.set_project_details(1, archived=1)
        db.close()
        stats = library.home_stats([library.summarize_project(p) for p in (first, archived, self.root / "gone.sqlite")])
        self.assertEqual((stats.projects, stats.matches_analysed, stats.ready_to_review), (1, 2, 1))
        self.assertAlmostEqual(stats.hours_processed, 300 / 3600)


class LibraryTests(TempFolderTest):
    def test_register_orders_by_recent_use_and_forget_clears_last(self):
        store = library.ProjectLibrary(self.root / "home")
        a, b = self.make_project(folder="a"), self.make_project(folder="b")
        store.register(a); store.register(b); store.register(a)
        self.assertEqual(store.paths(), [a.resolve(), b.resolve()])
        self.assertEqual(store.last(), a.resolve())
        store.forget(a)
        self.assertEqual(store.paths(), [b.resolve()])
        self.assertIsNone(store.last())
        moved = self.root / "moved.sqlite"
        store.replace(b, moved)
        self.assertEqual(store.paths(), [moved.resolve()])

    def test_damaged_library_entries_are_ignored(self):
        store = library.ProjectLibrary(self.root / "home")
        store.file.parent.mkdir(parents=True)
        store.file.write_text(json.dumps({"projects": [123, None, "", "x.sqlite"], "last": 5}), encoding="utf-8")
        self.assertEqual(store.paths(), [Path("x.sqlite")])
        self.assertIsNone(store.last())
        project = self.make_project()
        store.register(project)
        store.forget(project)
        self.assertEqual(store.paths(), [Path("x.sqlite")])

    def test_a_failed_duplicate_leaves_no_partial_file(self):
        from unittest.mock import patch
        source = self.make_project()
        target = self.root / "copy" / "control_vision.sqlite"
        target.parent.mkdir()
        with patch.object(ProjectDatabase, "set_video_details", side_effect=ValueError("bad camera view")):
            with self.assertRaises(ValueError):
                library.duplicate_project(source, target, "copy")
        self.assertFalse(target.exists())
        library.duplicate_project(source, target, "copy")  # A retry into the same folder works.
        self.assertEqual(library.summarize_project(target).name, "copy")

    def test_unique_names(self):
        self.assertEqual(library.unique_name("Meet", set()), "Meet")
        self.assertEqual(library.unique_name("Meet", {"meet", "Meet (2)"}), "Meet (3)")

    def test_duplicate_copies_videos_but_not_results(self):
        source = self.make_project(videos=2, ready=1, flagged=1)
        target = library.duplicate_project(source, self.root / "copy" / "control_vision.sqlite", "Meet 1 copy")
        copy = library.summarize_project(target)
        self.assertEqual((copy.name, copy.matches, copy.processed, copy.flagged), ("Meet 1 copy", 2, 0, 0))
        self.assertEqual(copy.event_code, "WASALK")
        with self.assertRaises(FileExistsError):
            library.duplicate_project(source, target, "again")

    def test_export_writes_a_readable_copy(self):
        source = self.make_project()
        target = library.export_project_file(source, self.root / "export.db")
        self.assertEqual(library.summarize_project(target).name, "Meet 1")
        with self.assertRaises(ValueError):
            library.export_project_file(source, source)


class ModelStoreTests(TempFolderTest):
    def write_model(self, name, metrics=None):
        path = self.root / f"{name}.onnx"
        path.write_bytes(b"onnx")
        if metrics is not None:
            (self.root / f"{name}.metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return path

    def test_metrics_come_only_from_a_sidecar(self):
        store = library.ModelStore(self.root / "home")
        bare = store.add_file(self.write_model("v1"))
        self.assertIsNone(bare.map50)
        self.assertEqual(store.active().id, bare.id)  # The first model becomes active.
        self.assertEqual(store.recommended()[0], None)

    def test_recommendation_prefers_robot_recall(self):
        store = library.ModelStore(self.root / "home")
        store.add_file(self.write_model("v2", {"map50": 0.9, "per_class": {"robot": {"recall": 0.7}}}))
        better = store.add_file(self.write_model("v3", {"map50": 0.8, "per_class": {"robot": {"recall": 0.9}}}))
        model, reason = store.recommended()
        self.assertEqual(model.id, better.id)
        self.assertIn("robot recall", reason)
        with self.assertRaises(KeyError):
            store.set_active("nope")
        with self.assertRaises(FileNotFoundError):
            store.add_file(self.root / "missing.onnx")


if __name__ == "__main__":
    unittest.main()
