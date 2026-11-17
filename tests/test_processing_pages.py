"""UI phase 3: processing queue, live view, match results and field map."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from test_setup_pages import CORNERS, SetupTest, qt_app, select_row
except ImportError:  # run as tests.test_processing_pages
    from tests.test_setup_pages import CORNERS, SetupTest, qt_app, select_row

PERIODS = {"auto": 30, "transition": 8, "teleop": 120}


class ReadyMatch(SetupTest):
    """A calibrated match with four robots (one absent) and stored positions."""

    def ready_match(self, name="Q1.mp4", processed=True):
        video, path = self.add_video(name)
        self.db.save_calibration(video, CORNERS, {"Hive": {"type": "scoring", "points": [[5, 5], [7, 5], [7, 7], [5, 7]]}})
        ids = self.db.save_identity(video, [
            {"team_number": 21229, "alliance": "red", "seed_x": 40, "seed_y": 60},
            {"team_number": 27393, "alliance": "red", "seed_x": 80, "seed_y": 60},
            {"team_number": 16655, "alliance": "blue", "seed_x": 120, "seed_y": 60},
            {"team_number": 35817, "alliance": "blue", "absent": True}])
        rows = [(ids[0], frame, frame / 8, 1 + frame * 0.1, 2.0, 0.9) for frame in range(0, 16)]
        rows += [(ids[0], frame, frame / 8, 6.0, 6.0, 0.9) for frame in range(40, 48)]   # after a 3 s gap
        rows += [(ids[2], frame, frame / 8, 10.0, 1 + frame * 0.05, 0.9) for frame in range(0, 20)]
        self.db.replace_tracks(rows, ids)
        if processed:
            self.db.set_analysis_status(video, "ready")
        return video, ids


class JobStorageTests(SetupTest):
    def test_jobs_keep_their_settings_order_and_survive_a_restart(self):
        first, _ = self.add_video("Q1.mp4")
        second, _ = self.add_video("Q2.mp4")
        ids = self.db.queue_jobs([first, second, first], 6, "yolo", "models/a.onnx")
        self.assertEqual(len(ids), 2)                      # A video waits once.
        self.assertEqual(self.db.queue_jobs([first], 8, "yolo", None), [])
        with self.assertRaises(ValueError):
            self.db.queue_jobs([second], 12, "yolo", None)
        self.assertTrue(self.db.move_job(ids[1], -1))
        self.assertFalse(self.db.move_job(ids[1], -1))     # Already first.
        self.assertEqual([int(job["id"]) for job in self.db.processing_jobs()], [ids[1], ids[0]])
        self.assertEqual(self.db.next_queued_job()["sample_fps"], 6)
        self.db.set_job_status(ids[1], "running")
        self.db.set_job_status(ids[0], "done")
        self.assertEqual(self.db.recover_interrupted_jobs(), 1)   # The app closed mid-run.
        self.assertEqual(self.db.next_queued_job()["id"], ids[1])
        self.db.set_job_status(ids[1], "failed", "Calibration missing — calibrate this match first.")
        self.assertEqual(self.db.clear_finished_jobs(), 1)
        self.assertEqual(self.db.retry_failed_jobs(), 1)
        self.assertEqual(self.db.processing_jobs()[0]["error"], None)


class RunLogTests(unittest.TestCase):
    def test_progress_and_log_lines_are_read_not_guessed(self):
        from control_vision.app.runlog import frames_progress, is_id_switch, progress_time, sampled_frame, time_left
        self.assertEqual(frames_progress(10, "Processed 12 of 120 sampled frames at 1.5 s"), (12, 120))
        self.assertEqual(frames_progress(25, "Processed 30 sampled frames · elapsed 4s"), (30, 120))
        self.assertIsNone(frames_progress(0, "Loading model"))
        # The pipeline formats counts with thousands separators.
        self.assertEqual(frames_progress(25, "Processed 1,234 of 5,000 sampled frames at 205.6s"), (1234, 5000))
        self.assertEqual(frames_progress(50, "Processed 1,500 sampled frames"), (1500, 3000))
        self.assertEqual(progress_time("Processed 12 of 120 sampled frames at 1.5 s"), 1.5)
        self.assertEqual(sampled_frame("Sampled frame 48 (1.600s): 3 robot detections; 0 scoring elements in ROI"),
                         (48, 1.6, 3))
        self.assertTrue(is_id_switch("Recovered robot 2 at 41.20s: track 7 -> new track 11 after 1.10s missing"))
        self.assertTrue(is_id_switch("Re-seeded robot 3 at 2.0s"))
        self.assertFalse(is_id_switch("Robot 2 not re-linked: two candidates"))
        self.assertIsNone(time_left(5, 0.5))
        self.assertEqual(time_left(60, 25), 180)

    def test_phase_changes_are_reported_and_a_skipped_transition_is_flagged(self):
        from control_vision.app.runlog import PhaseTracker
        phases = PhaseTracker(PERIODS)
        self.assertEqual(phases.update(10)[0], "AUTO")
        phase, _left, change, skipped = phases.update(31)
        self.assertEqual((phase, change, skipped), ("TRANSITION", "Phase changed: AUTO → TRANSITION", False))
        jump = PhaseTracker(PERIODS)
        jump.update(5)
        phase, _left, change, skipped = jump.update(50)
        self.assertEqual(phase, "TELEOP")
        self.assertTrue(skipped)
        self.assertIn("TRANSITION missing", change)


class QueuePageTests(SetupTest):
    def page(self):
        from control_vision.app.pages.queue_page import QueuePage
        page = QueuePage(lambda: (8, "yolo", str(self.root / "model.onnx")))
        page.set_project(self.db, self.project)
        return page

    def test_queue_rows_actions_and_fixes(self):
        page = self.page()
        self.assertFalse(page.empty.isHidden())
        video, _ = self.add_video()
        [job] = page.queue([video], 6)
        self.assertEqual(page.table.rowCount(), 1)
        self.assertEqual((page.table.item(0, 3).text(), page.table.item(0, 4).text(), page.table.item(0, 5).text()),
                         ("model.onnx", "6", "Queued"))
        self.assertTrue(page.start_btn.isEnabled())
        self.db.set_job_status(job, "failed", "Calibration missing — calibrate this match first.")
        page.refresh()
        select_row(page.table, 0)
        self.assertEqual(page.open_btn.text(), "Calibrate…")
        self.assertIn("Calibration missing", page.error.text())
        with patch.object(page, "open_calibrate") as fix:
            page.open_selected()
        fix.emit.assert_called_once_with(video)
        page.retry_failed()
        self.assertEqual(page.table.item(0, 5).text(), "Queued")

    def test_running_job_shows_progress_and_resume_prompt(self):
        page = self.page()
        video, _ = self.add_video()
        [job] = page.queue([video])
        self.db.set_job_status(job, "running")
        page.set_runner_state(job, False)
        page.set_progress(job, 40, 90)
        self.assertEqual(page.table.cellWidget(0, 6).value(), 40)
        self.assertEqual(page.table.item(0, 7).text(), "01:30")
        select_row(page.table, 0)
        self.assertEqual(page.open_btn.text(), "Open live view")
        self.assertFalse(page.remove_btn.isEnabled())
        with patch.object(page, "cancel_requested") as cancel:
            page._cancel_selected()
        cancel.emit.assert_called_once_with(job)
        page.show_resume_prompt(3)
        self.assertIn("3 job(s) were queued when the app closed", page.banner.text())


class LivePageTests(unittest.TestCase):
    def setUp(self):
        self.app = qt_app()

    def test_live_view_reads_the_run(self):
        from control_vision.app.pages.live_page import PREVIEW_NOTE, LivePage
        clock = iter([0.0, 10.0, 10.0, 20.0, 20.0, 30.0])
        page = LivePage(PERIODS, clock=lambda: next(clock, 30.0))
        self.assertFalse(page.idle.isHidden())
        page.start_run(4, Path("missing.mp4"), 8)
        page.set_progress("Detect and track", 10, "Processed 24 of 240 sampled frames at 3.0 s")
        self.assertEqual(page.values["frames"].text(), "24 / 240")
        self.assertTrue(page.values["phase"].text().startswith("AUTO"))
        page.add_line("INFO", "Sampled frame 300 (31.000s): 3 robot detections; 0 scoring elements in ROI")
        self.assertEqual(page.values["detections"].text(), "3")
        self.assertTrue(page.values["phase"].text().startswith("TRANSITION"))
        page.add_line("INFO", "Recovered robot 2 at 31.20s: track 7 -> new track 11 after 1.10s missing")
        self.assertEqual(page.values["switches"].text(), "1")
        log = [page.log.item(i).text() for i in range(page.log.count())]
        self.assertIn("Phase changed: AUTO → TRANSITION", log)
        self.assertTrue(any(line.startswith("Recovered robot 2") for line in log))
        self.assertIn("without boxes", PREVIEW_NOTE)
        page.failed("Processing was cancelled.")
        self.assertEqual(page.state, "cancelled")
        self.assertFalse(page.retry_btn.isHidden())

    def test_stop_asks_first(self):
        from control_vision.app.pages import live_page
        page = live_page.LivePage(PERIODS)
        page.start_run(4, Path("missing.mp4"), 8)
        with patch.object(live_page.QMessageBox, "question", return_value=live_page.QMessageBox.StandardButton.No), \
                patch.object(page, "stop_requested") as stop:
            page.confirm_stop()
        stop.emit.assert_not_called()
        with patch.object(live_page.QMessageBox, "question", return_value=live_page.QMessageBox.StandardButton.Yes), \
                patch.object(page, "stop_requested") as stop:
            page.confirm_stop()
        stop.emit.assert_called_once()


class ResultsTests(ReadyMatch):
    def test_ours_and_official_side_by_side(self):
        from control_vision.app.pages.results_page import ResultsPage, official_check
        self.assertEqual(official_check(10, None), ("Official score not entered yet", "pending"))
        self.assertEqual(official_check(52, 50)[1], "success")
        self.assertIn("Differs from official by 20%", official_check(40, 50)[0])
        video, ids = self.ready_match()
        for robot, points, status in ((ids[0], 10, "confirmed"), (ids[2], 6, "confirmed"), (ids[1], 4, "flagged")):
            self.db.connection.execute(
                "INSERT INTO scoring_events(robot_id,video_id,t_s,points,status) VALUES (?,?,?,?,?)",
                (robot, video, 12.0, points, status))
        self.db.connection.commit()
        page = ResultsPage()
        page.set_project(self.db, self.project)
        self.assertEqual(page.cards["red"]["ours"].text(), "10")       # The flagged event is not counted.
        self.assertEqual(page.cards["blue"]["ours"].text(), "6")
        self.assertIn("Not reviewed yet", page.review_badge.text())
        self.assertEqual(page.cards["red"]["check"].text(), "Official score not entered yet")
        robot_texts = [page.robot_row.itemAt(i).widget().findChildren(type(page.header)) for i in range(4)]
        self.assertIn("Absent · 0 points", [label.text() for label in robot_texts[3]])
        page.cards["red"]["official"].setValue(10)
        page.cards["blue"]["official"].setValue(30)
        page.save_official()
        row = self.db.connection.execute("SELECT official_red,official_blue FROM videos WHERE id=?", (video,)).fetchone()
        self.assertEqual(tuple(row), (10, 30))
        self.assertIn("Within ±10%", page.cards["red"]["check"].text())
        self.assertIn("Differs from official by 80%", page.cards["blue"]["check"].text())

    def test_unprocessed_match_says_so(self):
        from control_vision.app.pages.results_page import ResultsPage
        self.ready_match(processed=False)
        page = ResultsPage()
        page.set_project(self.db, self.project)
        self.assertFalse(page.not_processed.isHidden())
        self.assertTrue(page.results.isHidden())


class FieldMapTests(ReadyMatch):
    def test_gaps_stay_gaps(self):
        from control_vision.app.pages.fieldmap_page import RobotTrail, heat_counts
        trail = RobotTrail(1, "21229", "red", samples=[(0.0, 1, 1), (0.5, 2, 1), (4.0, 6, 6), (4.5, 6, 7)])
        self.assertEqual(len(trail.segments()), 2)          # 3.5 s apart: a gap, not a line.
        self.assertEqual(trail.gaps(), 1)
        self.assertAlmostEqual(trail.distance_ft(), 2.0)    # The jump across the gap is not driven distance.
        self.assertEqual(len(trail.segments(until_s=1.0)), 1)
        self.assertEqual(sum(map(sum, heat_counts([trail]))), 4)

    def test_page_states_and_click_to_seek(self):
        from control_vision.app.pages.fieldmap_page import FieldMapPage
        page = FieldMapPage()
        uncalibrated, _ = self.add_video("Q0.mp4")
        page.set_project(self.db, self.project)
        page.refresh_videos(select=uncalibrated)
        self.assertEqual(page.empty.title.text(), "No calibration — calibrate first")
        video, ids = self.ready_match()
        page.refresh_videos(select=video)
        self.assertTrue(page.empty.isHidden())
        legend = [page.legend.itemAt(i).widget().text() for i in range(page.legend.count())]
        self.assertTrue(legend[0].startswith("21229 · Red"))
        self.assertIn("1 gap(s)", legend[0])
        self.assertIn("no positions", legend[3])
        self.assertFalse(page.balls.isEnabled())
        page.resize(1000, 700)
        page.map.resize(600, 600)
        target = page.map.to_widget(6.0, 6.0)
        self.assertEqual(page.map.nearest_point(target), (ids[0], 5.0))
        page._point_clicked(ids[0], 5.0)
        self.assertEqual(page.scrubber.value(), 50)
        self.assertIn("team 21229", page.time_label.text())
        not_processed, _ = self.ready_match("Q2.mp4", processed=False)
        page.refresh_videos(select=not_processed)
        self.assertEqual(page.empty.title.text(), "Processing not finished")


class WindowQueueTests(ReadyMatch):
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

    def open(self):
        self.window.db, self.window.project_id = self.db, self.project
        self.window.refresh_home()

    def test_jobs_that_cannot_run_fail_with_the_fix_and_the_queue_moves_on(self):
        from control_vision.app import main_window as mw
        missing, _ = self.add_video("Q0.mp4")
        video, _ = self.ready_match(processed=False)
        self.open()
        model = self.root / "model.onnx"; model.write_bytes(b"x")
        first, second = self.db.queue_jobs([missing, video], 6, "yolo", str(model))
        window = self.window
        window.sample_rate.setCurrentText("8")
        started = []
        with patch.object(window, "process", side_effect=lambda: (started.append(window.sample_rate.currentText()),
                                                                 setattr(window, "_processing_active", True))):
            window.start_queue()
            self.app.processEvents()
        jobs = {int(job["id"]): job for job in self.db.processing_jobs()}
        self.assertEqual(jobs[first]["status"], "failed")
        self.assertTrue(jobs[first]["error"].startswith("Calibration missing"))
        self.assertEqual(jobs[second]["status"], "running")
        self.assertEqual(started, ["6"])                      # The job's own sample rate.
        window.processing_failed("Processing was cancelled.")
        self.app.processEvents()
        self.assertEqual(self.db.processing_jobs()[1]["status"], "cancelled")
        self.assertEqual(window.sample_rate.currentText(), "8")   # The user's setting is back.
        self.assertIsNone(window._queue_job)
        self.assertEqual(window.live_page.state, "idle")      # process() was replaced, so no run started here.
        self.assertIn((mw.QUEUE_TAB, "Processing queue"), [(tab, label) for tab, label in mw.NAV_ENTRIES if tab != "section"])

    def test_stopping_a_queued_run_pauses_the_queue(self):
        first_video, _ = self.ready_match("Q1.mp4", processed=False)
        second_video, _ = self.ready_match("Q2.mp4", processed=False)
        self.open()
        model = self.root / "model.onnx"; model.write_bytes(b"x")
        first, second = self.db.queue_jobs([first_video, second_video], 8, "yolo", str(model))
        window = self.window
        started = []
        with patch.object(window, "process", side_effect=lambda: (started.append(window.video_id),
                                                                 setattr(window, "_processing_active", True))):
            window.start_queue()
            self.app.processEvents()
            window.processing_failed("Processing was cancelled.")
            self.app.processEvents()
            jobs = {int(job["id"]): job["status"] for job in self.db.processing_jobs()}
            self.assertEqual(jobs, {first: "cancelled", second: "queued"})
            self.assertTrue(window._queue_paused)
            self.assertEqual(len(started), 1)               # The next match did not start.
            window.start_queue()                            # Start queue carries on with the next one.
            self.app.processEvents()
        self.assertEqual(len(started), 2)
        self.assertEqual(window._queue_job, second)

    def test_the_queue_pauses_and_says_so_while_another_run_is_active(self):
        video, _ = self.ready_match(processed=False)
        self.open()
        model = self.root / "model.onnx"; model.write_bytes(b"x")
        [job] = self.db.queue_jobs([video], 8, "yolo", str(model))
        window = self.window
        window._processing_active = True                    # A single-match run is going.
        try:
            with patch.object(window, "process") as process:
                window.start_queue()
                self.app.processEvents()
            process.assert_not_called()
            self.assertTrue(window._queue_paused)
            self.assertIn("another run is in progress", window.progress_label.text())
            self.assertEqual(self.db.processing_jobs()[0]["status"], "queued")
        finally:
            window._processing_active = False

    def test_a_job_whose_start_raises_fails_and_pauses_the_queue(self):
        first_video, _ = self.ready_match("Q1.mp4", processed=False)
        second_video, _ = self.ready_match("Q2.mp4", processed=False)
        self.open()
        model = self.root / "model.onnx"; model.write_bytes(b"x")
        first, second = self.db.queue_jobs([first_video, second_video], 6, "yolo", str(model))
        window = self.window
        window.sample_rate.setCurrentText("8")
        with patch.object(window, "process", side_effect=OSError("run log is read-only")) as process:
            window.start_queue()
            self.app.processEvents()
        self.assertEqual(process.call_count, 1)
        jobs = {int(job["id"]): job for job in self.db.processing_jobs()}
        self.assertEqual(jobs[first]["status"], "failed")
        self.assertIn("run log is read-only", jobs[first]["error"])
        self.assertEqual(jobs[second]["status"], "queued")
        self.assertIsNone(window._queue_job)
        self.assertTrue(window._queue_paused)
        self.assertEqual(window.sample_rate.currentText(), "8")   # The user's setting is back.
        self.assertIn("Processing could not start", window.progress_label.text())

    def test_a_finished_job_is_marked_done(self):
        video, ids = self.ready_match(processed=False)
        self.open()
        model = self.root / "model.onnx"; model.write_bytes(b"x")
        [job] = self.db.queue_jobs([video], 8, "yolo", str(model))
        window = self.window
        with patch.object(window, "process", side_effect=lambda: setattr(window, "_processing_active", True)):
            window.start_queue()
        with patch.object(window, "_store_processing_results"):
            window.processing_done((([(ids[0], 0, 0.0, 1.0, 1.0, 0.9)], [], []), "yolo"))
        self.app.processEvents()
        self.assertEqual(self.db.processing_jobs()[0]["status"], "done")

    def test_opening_a_project_offers_to_resume(self):
        video, _ = self.ready_match(processed=False)
        [job] = self.db.queue_jobs([video], 8, "yolo", None)
        self.db.set_job_status(job, "running")              # The app closed mid-run.
        self.open()
        self.assertEqual(self.db.processing_jobs()[0]["status"], "queued")
        self.assertIn("1 job(s) were queued", self.window.queue_page.banner.text())


if __name__ == "__main__":
    unittest.main()
