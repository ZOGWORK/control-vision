"""Processing queue: matches analysed one after another in the background (guide page 13).

The page shows and edits the queue stored in the project; the window runs the jobs with
the same processing as Set up and process, one at a time, with each job's own settings.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from control_vision.app import widgets
from control_vision.app.media import format_clock

COLUMNS = ("Match", "Event · match", "Length", "Model", "Fps", "Status", "Progress", "Time left")
STATUS_TEXT = {"queued": ("Queued", "muted"), "running": ("Running", "info"), "done": ("✓ Done", "success"),
               "failed": ("✕ Failed", "danger"), "cancelled": ("Cancelled", "pending")}
MODE_TEXT = {"yolo": "YOLO", "seeded_optical_flow": "Seeded preview"}
# Failures the user fixes on a setup page; they are never retried automatically.
FIXES = {"Calibration missing": "calibrate", "Robot identities missing": "identity"}


def fix_for(error: str | None) -> str | None:
    return next((page for prefix, page in FIXES.items() if (error or "").startswith(prefix)), None)


def model_text(mode: str, model_path: str | None) -> str:
    if mode == "yolo":
        return Path(model_path).name if model_path else "YOLO (no model set)"
    return MODE_TEXT.get(mode, mode)


def resource_text() -> str:
    """CPU use, and whether a YOLO training run is sharing it; empty when psutil is not available."""
    try:
        import psutil
    except ImportError:
        return ""
    text = f"CPU {psutil.cpu_percent(interval=None):.0f}%"
    try:
        training = any("train_yolo" in " ".join(process.info.get("cmdline") or [])
                       for process in psutil.process_iter(["cmdline"]))
    except (psutil.Error, OSError):
        training = False
    return text + (" · Sharing CPU with training, slower than usual" if training else "")


class AddMatchesDialog(QDialog):
    """Pick the project's matches that are not processed or queued yet, and the sample rate."""

    def __init__(self, videos, sample_fps: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add matches to the queue")
        layout = QVBoxLayout(self)
        layout.addWidget(widgets.label("Matches not processed yet. Each job keeps the sample rate and model chosen now.",
                                       role="muted", wrap=True))
        self.list = QListWidget()
        self.list.setAccessibleName("Matches to add")
        for row in videos:
            item = QListWidgetItem(f"{Path(row['path']).name}" + (f" · {row['match_number']}" if row["match_number"] else ""))
            item.setData(Qt.ItemDataRole.UserRole, int(row["id"]))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list.addItem(item)
        layout.addWidget(self.list, 1)
        if not videos:
            layout.addWidget(widgets.label("Every match in this project is processed or already queued.", wrap=True))
        row = QHBoxLayout()
        fps_label = QLabel("&Sample rate (frames per second)")
        self.fps = QComboBox()
        for value in range(5, 11):
            self.fps.addItem(str(value), value)
        self.fps.setCurrentIndex(max(0, self.fps.findData(int(sample_fps))))
        fps_label.setBuddy(self.fps)
        row.addWidget(fps_label); row.addWidget(self.fps); row.addStretch(1)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add to queue")
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def chosen(self) -> list[int]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]


class QueuePage(QWidget):
    start_requested = Signal()
    pause_requested = Signal()
    cancel_requested = Signal(int)        # job id
    open_live = Signal()
    open_results = Signal(int)            # video id
    open_calibrate = Signal(int)
    open_identity = Signal(int)
    new_project = Signal()

    def __init__(self, settings=None):
        """``settings`` returns the defaults for new jobs: (sample fps, tracking mode, model path)."""
        super().__init__()
        self.setObjectName("Page")
        self.settings = settings or (lambda: (8, "yolo", None))
        self.db = None
        self.project_id = None
        self.running_job: int | None = None
        self.paused = True
        self._progress: dict[int, tuple[float, float | None]] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        self.no_project = widgets.EmptyState("Open a project to process matches",
                                             "The queue belongs to a project. Create one, or open one from Projects.",
                                             "+ New project", self.new_project.emit)
        outer.addWidget(self.no_project, 1)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        outer.addWidget(self.body, 1)

        top = QHBoxLayout()
        self.add_btn = widgets.button("Add matches…", on_click=self.add_matches)
        self.start_btn = widgets.button("Start queue", "primary", on_click=self.start_requested.emit)
        self.pause_btn = widgets.button("Pause all", on_click=self.pause_requested.emit,
                                        tooltip="The running match finishes; the next one does not start.")
        self.clear_btn = widgets.button("Clear finished", on_click=self.clear_finished)
        self.retry_btn = widgets.button("Retry failed", on_click=self.retry_failed)
        for control in (self.add_btn, self.start_btn, self.pause_btn, self.clear_btn, self.retry_btn):
            top.addWidget(control)
        top.addStretch(1)
        self.resources = widgets.label("", role="muted")
        self.resources.setAccessibleName("Computer load")
        top.addWidget(self.resources)
        body.addLayout(top)

        self.banner = widgets.label("", tone="pending", wrap=True)
        self.banner.hide()
        body.addWidget(self.banner)

        self.empty = widgets.EmptyState("No jobs — add a match to process",
                                        "Matches need a calibration and at least one robot marker to be processed.",
                                        "Add matches…", self.add_matches)
        body.addWidget(self.empty, 1)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setAccessibleName("Processing queue")
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(6, 160)
        self.table.itemSelectionChanged.connect(self._sync_actions)
        self.table.cellDoubleClicked.connect(lambda *_: self.open_selected())
        body.addWidget(self.table, 1)
        self.error = widgets.label("", tone="danger", wrap=True)
        self.error.hide()
        body.addWidget(self.error)

        actions = QHBoxLayout()
        self.up_btn = widgets.button("Move up", on_click=lambda: self.move_selected(-1))
        self.down_btn = widgets.button("Move down", on_click=lambda: self.move_selected(1))
        self.remove_btn = widgets.button("Remove", on_click=self.remove_selected)
        self.cancel_btn = widgets.button("Cancel", "danger", on_click=self._cancel_selected,
                                         tooltip="Stops after the current frame; nothing from a cancelled run is saved.")
        self.open_btn = widgets.button("Open", on_click=self.open_selected)
        for control in (self.up_btn, self.down_btn, self.remove_btn, self.cancel_btn):
            actions.addWidget(control)
        actions.addStretch(1)
        actions.addWidget(self.open_btn)
        body.addLayout(actions)

        self._load_timer = QTimer(self)
        self._load_timer.setInterval(5000)
        self._load_timer.timeout.connect(lambda: self.resources.setText(resource_text()))
        self.no_project.show(); self.body.hide()

    # ----- project -----

    def set_project(self, db, project_id) -> None:
        self.db, self.project_id = db, project_id
        has = db is not None and project_id is not None
        self.no_project.setVisible(not has)
        self.body.setVisible(has)
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.resources.setText(resource_text())
        self._load_timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._load_timer.stop()

    def jobs(self) -> list:
        return list(self.db.processing_jobs()) if self.db is not None else []

    def show_resume_prompt(self, waiting: int) -> None:
        if waiting:
            self.banner.setText(f"⚑ {waiting} job(s) were queued when the app closed. Start queue to resume, "
                                "or Remove them.")
        self.banner.setVisible(bool(waiting))

    # ----- state from the runner -----

    def set_runner_state(self, running_job: int | None, paused: bool) -> None:
        if running_job is not None and running_job != self.running_job:
            self._progress.pop(int(running_job), None)  # A retried job starts again from 0%.
        self.running_job, self.paused = running_job, paused
        if running_job is not None:
            self.banner.hide()
        self.refresh()

    def set_progress(self, job_id: int, percent: float, seconds_left: float | None) -> None:
        self._progress[int(job_id)] = (percent, seconds_left)
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == int(job_id):
                self._show_progress(row, int(job_id), "running")

    # ----- table -----

    def refresh(self) -> None:
        selected = self.selected_job()
        jobs = self.jobs()
        self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            status, tone = STATUS_TEXT.get(job["status"], (job["status"], "muted"))
            event = " · ".join(part for part in (job["event_code"], job["match_number"]) if part) or "—"
            values = (Path(job["path"]).name, event, format_clock(job["duration_s"] or 0),
                      model_text(job["tracking_mode"], job["model_path"]), str(job["sample_fps"]), status)
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, int(job["id"]))
                if column == 0:
                    item.setToolTip(job["path"])
                if column == 5:
                    item.setForeground(_tone_brush(tone))
                    if job["error"]:
                        item.setToolTip(job["error"])
                self.table.setItem(row, column, item)
            self._show_progress(row, int(job["id"]), job["status"])
        if selected is not None:
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == selected["id"]:
                    self.table.selectRow(row)
        self.empty.setVisible(not jobs)
        self.table.setVisible(bool(jobs))
        waiting = any(job["status"] == "queued" for job in jobs)
        running = self.running_job is not None
        widgets.disable_with_reason(self.start_btn, waiting and (self.paused or not running),
                                    "Nothing is waiting, or the queue is already running")
        self.start_btn.setText("Resume queue" if waiting and self.paused and any(
            job["status"] in ("done", "failed", "cancelled") for job in jobs) else "Start queue")
        widgets.disable_with_reason(self.pause_btn, not self.paused, "The queue is not running")
        widgets.disable_with_reason(self.clear_btn, any(job["status"] in ("done", "cancelled") for job in jobs),
                                    "No finished jobs")
        widgets.disable_with_reason(self.retry_btn, any(job["status"] in ("failed", "cancelled") for job in jobs),
                                    "No failed jobs")
        self._sync_actions()

    def _show_progress(self, row: int, job_id: int, status: str) -> None:
        if status == "running":
            percent, left = self._progress.get(job_id, (0.0, None))
            bar = self.table.cellWidget(row, 6)
            if not isinstance(bar, QProgressBar):
                bar = QProgressBar(); bar.setRange(0, 100); bar.setAccessibleName("Job progress")
                self.table.setCellWidget(row, 6, bar)
            bar.setValue(max(0, min(100, round(percent))))
            bar.setFormat(f"{percent:.0f}%")
            self.table.setItem(row, 7, QTableWidgetItem(format_clock(left) if left is not None else "Estimating…"))
        else:
            if self.table.cellWidget(row, 6) is not None:
                self.table.removeCellWidget(row, 6)
            self.table.setItem(row, 6, QTableWidgetItem("100%" if status == "done" else ""))
            self.table.setItem(row, 7, QTableWidgetItem(""))

    def selected_job(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or self.db is None:
            return None
        job_id = self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        return next((job for job in self.jobs() if int(job["id"]) == int(job_id)), None)

    def _sync_actions(self) -> None:
        job = self.selected_job()
        status = job["status"] if job else None
        widgets.disable_with_reason(self.up_btn, job is not None, "Select a job")
        widgets.disable_with_reason(self.down_btn, job is not None, "Select a job")
        widgets.disable_with_reason(self.remove_btn, job is not None and status != "running",
                                    "Select a job that is not running")
        widgets.disable_with_reason(self.cancel_btn, status == "running", "Select the running job")
        fix = fix_for(job["error"]) if job and status == "failed" else None
        label = {"running": "Open live view", "done": "Open results"}.get(status)
        if fix == "calibrate":
            label = "Calibrate…"
        elif fix == "identity":
            label = "Assign robots…"
        self.open_btn.setText(label or "Open")
        widgets.disable_with_reason(self.open_btn, label is not None, "Select a running, finished or fixable job")
        if job is not None and status == "failed" and job["error"]:
            self.error.setText(f"✕ {job['error']}")
            self.error.show()
        else:
            self.error.hide()

    # ----- actions -----

    def add_matches(self) -> None:
        if self.db is None:
            return
        queued = {int(job["video_id"]) for job in self.jobs() if job["status"] in ("queued", "running")}
        videos = [row for row in self.db.connection.execute(
            "SELECT id,path,match_number,analysis_status FROM videos WHERE project_id=? ORDER BY id", (self.project_id,))
            if row["analysis_status"] != "ready" and int(row["id"]) not in queued]
        fps, mode, model = self.settings()
        dialog = AddMatchesDialog(videos, fps, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.queue(dialog.chosen(), dialog.fps.currentData())

    def queue(self, video_ids, sample_fps: int | None = None) -> list[int]:
        fps, mode, model = self.settings()
        ids = self.db.queue_jobs(video_ids, sample_fps or fps, mode, model)
        self.refresh()
        return ids

    def move_selected(self, offset: int) -> None:
        job = self.selected_job()
        if job is not None and self.db.move_job(int(job["id"]), offset):
            self.refresh()

    def remove_selected(self) -> None:
        job = self.selected_job()
        if job is not None and job["status"] != "running":
            self.db.remove_jobs([int(job["id"])])
            self.refresh()

    def clear_finished(self) -> None:
        self.db.clear_finished_jobs()
        self.refresh()

    def retry_failed(self) -> None:
        self.db.retry_failed_jobs()
        self.refresh()

    def _cancel_selected(self) -> None:
        job = self.selected_job()
        if job is not None and job["status"] == "running":
            self.cancel_requested.emit(int(job["id"]))

    def open_selected(self) -> None:
        job = self.selected_job()
        if job is None:
            return
        if job["status"] == "running":
            self.open_live.emit()
        elif job["status"] == "done":
            self.open_results.emit(int(job["video_id"]))
        elif job["status"] == "failed":
            fix = fix_for(job["error"])
            if fix == "calibrate":
                self.open_calibrate.emit(int(job["video_id"]))
            elif fix == "identity":
                self.open_identity.emit(int(job["video_id"]))


def _tone_brush(tone: str):
    from PySide6.QtGui import QBrush, QColor
    from control_vision.app.frameview import tokens
    colours = tokens()
    return QBrush(QColor({"success": colours.success_fg, "pending": colours.pending_fg, "danger": colours.danger_fg,
                          "info": colours.accent}.get(tone, colours.text_muted)))
