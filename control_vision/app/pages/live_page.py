"""Processing live: watch the match being analysed (guide page 14).

Everything shown comes from the run itself: its progress reports and its log. The
pipeline does not hand boxes or positions to the app until it finishes, so the
picture is the video at the time being processed, without boxes, and the page
says so. Results are final only after processing and Review.
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QVBoxLayout, QWidget,
)

from control_vision.app import theme, widgets
from control_vision.app.frameview import FrameView, tokens
from control_vision.app.media import format_clock, read_frame
from control_vision.app.runlog import PhaseTracker, frames_progress, is_id_switch, progress_time, sampled_frame, time_left

PREVIEW_NOTE = ("Preview — the frame being analysed, without boxes: robot boxes and IDs are not available while a "
                "run is in progress in this build. Results are final only after processing and Review.")
PHASE_NOTE = "Phase from video time; assumes the video starts when the match starts."
FRAME_EVERY_S = 1.0     # how often the picture follows the run


class LivePage(QWidget):
    stop_requested = Signal()
    open_queue = Signal()
    open_results = Signal(int)    # video id
    retry_requested = Signal(int)  # video id

    def __init__(self, periods: dict | None = None, clock=time.monotonic):
        super().__init__()
        self.setObjectName("Page")
        self.periods = periods or {"auto": 30, "transition": 8, "teleop": 120}
        self.clock = clock
        self.video_id: int | None = None
        self.video_path: Path | None = None
        self.state = "idle"
        self._reset_counters()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        self.idle = widgets.EmptyState("No match is processing",
                                       "Start a match from Set up and process, or start the Processing queue.",
                                       "Open the processing queue", self.open_queue.emit)
        outer.addWidget(self.idle, 1)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        outer.addWidget(self.body, 1)
        head = QHBoxLayout()
        self.title = widgets.label("", role="section")
        head.addWidget(self.title, 1)
        self.state_label = widgets.label("")
        head.addWidget(self.state_label)
        self.stop_btn = widgets.button("Stop processing", "danger", on_click=self.confirm_stop)
        self.results_btn = widgets.button("Open results", "primary",
                                          on_click=lambda: self.open_results.emit(int(self.video_id)))
        self.retry_btn = widgets.button("Retry", on_click=lambda: self.retry_requested.emit(int(self.video_id)))
        for control in (self.stop_btn, self.results_btn, self.retry_btn):
            head.addWidget(control)
        body.addLayout(head)

        middle = QHBoxLayout()
        left = QVBoxLayout()
        self.view = FrameView("Waiting for the first frame…")
        self.view.setAccessibleName("Frame being processed")
        left.addWidget(self.view, 1)
        left.addWidget(widgets.label(PREVIEW_NOTE, role="muted", wrap=True))
        middle.addLayout(left, 3)

        panel = widgets.card()
        grid = QGridLayout(panel)
        grid.setContentsMargins(16, 12, 16, 12)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setAccessibleName("Run progress")
        grid.addWidget(self.progress, 0, 0, 1, 2)
        self.values: dict[str, object] = {}
        for row, (key, text) in enumerate((("frames", "Sampled frames"), ("rate", "Frames per second"),
                                           ("time", "Video time"), ("left", "Time left"), ("phase", "Phase"),
                                           ("detections", "Robot detections, this frame"),
                                           ("switches", "ID switches")), start=1):
            grid.addWidget(widgets.label(text, role="muted"), row, 0)
            value = widgets.label("—")
            value.setAccessibleName(text)
            grid.addWidget(value, row, 1)
            self.values[key] = value
        grid.addWidget(widgets.label(PHASE_NOTE, role="muted", wrap=True), 8, 0, 1, 2)
        grid.setRowStretch(9, 1)
        middle.addWidget(panel, 2)
        body.addLayout(middle, 3)

        body.addWidget(widgets.section_label("Live log"))
        self.log = QListWidget()
        self.log.setAccessibleName("Live log")
        body.addWidget(self.log, 2)
        self._show_state()

    def _reset_counters(self) -> None:
        self.started_at = self.clock()
        self.frames_done = 0
        self.id_switches = 0
        self.last_frame_shown = -FRAME_EVERY_S
        self.phases = PhaseTracker(self.periods)

    # ----- a run -----

    def start_run(self, video_id: int, path: Path, sample_fps: float) -> None:
        self.video_id, self.video_path, self.state = int(video_id), Path(path), "running"
        self._reset_counters()
        self.sample_fps = float(sample_fps)
        self.log.clear()
        self.progress.setValue(0)
        for value in self.values.values():
            value.setText("—")
        self.title.setText(f"{self.video_path.name} · {sample_fps:g} fps")
        self.view.set_frame(None)
        self.add_line("INFO", f"Processing {self.video_path.name} at {sample_fps:g} sampled frames per second.")
        self._show_state()

    def set_progress(self, stage: str, percent: float, detail: str) -> None:
        if self.state != "running":
            return
        self.progress.setValue(max(0, min(100, round(percent))))
        self.progress.setFormat(f"{stage} — {percent:.0f}%")
        frames = frames_progress(percent, detail)
        if frames:
            done, total = frames
            self.frames_done = done
            self.values["frames"].setText(f"{done:,} / {total:,}")
            elapsed = max(1e-6, self.clock() - self.started_at)
            self.values["rate"].setText(f"{done / elapsed:.1f}")
        left = time_left(self.clock() - self.started_at, percent)
        self.values["left"].setText(format_clock(left) if left is not None else "Estimating…")
        seconds = progress_time(detail)
        if seconds is not None:
            self._at_time(seconds)

    def add_line(self, level: str, message: str) -> None:
        frame = sampled_frame(message)
        if frame is not None:
            # One line per sampled frame: shown in the panel, not the log.
            _number, seconds, detections = frame
            self.values["detections"].setText(str(detections))
            self._at_time(seconds)
            return
        warning = level in ("WARNING", "ERROR")
        if is_id_switch(message):
            self.id_switches += 1
            self.values["switches"].setText(str(self.id_switches))
            warning = True
        self._log(message, "danger" if level == "ERROR" else "pending" if warning else None)

    def _at_time(self, seconds: float) -> None:
        self.values["time"].setText(format_clock(seconds))
        phase, remaining, change, skipped = self.phases.update(seconds)
        self.values["phase"].setText(f"{phase} · {format_clock(remaining)} left" if phase != "POST-MATCH" else phase)
        theme.set_tone(self.values["phase"], "danger" if skipped else None)
        if change:
            self._log(change, "danger" if skipped else None)
        if self.video_path is not None and seconds - self.last_frame_shown >= FRAME_EVERY_S:
            self.last_frame_shown = seconds
            frame = read_frame(self.video_path, seconds)
            if frame is not None:
                self.view.set_frame(frame)

    def _log(self, message: str, tone: str | None) -> None:
        item = QListWidgetItem(message)
        if tone:
            colours = tokens()
            item.setForeground(QBrush(QColor(colours.danger_fg if tone == "danger" else colours.pending_fg)))
        self.log.addItem(item)
        self.log.scrollToBottom()

    def finished(self) -> None:
        if self.state != "running":
            return
        self.state = "finished"
        self.progress.setValue(100)
        self._log(f"Processing finished. ID switches logged: {self.id_switches}.", None)
        self._show_state()

    def failed(self, message: str) -> None:
        if self.state != "running":
            return
        self.state = "cancelled" if "cancelled" in message.lower() else "failed"
        self._log(message, "danger")
        self._show_state()

    def confirm_stop(self) -> None:
        if self.state != "running":
            return
        answer = QMessageBox.question(self, "Stop processing",
                                      "Stop processing this match? Nothing from a stopped run is saved.",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.stop_requested.emit()
            self._log("Stopping after the current frame…", "pending")

    def _show_state(self) -> None:
        self.idle.setVisible(self.state == "idle")
        self.body.setVisible(self.state != "idle")
        text, tone = {"running": ("Processing…", None),
                      "finished": ("✓ Processing finished — open Results", "success"),
                      "failed": ("✕ Processing failed", "danger"),
                      "cancelled": ("Processing stopped — nothing was saved", "pending")}.get(self.state, ("", None))
        self.state_label.setText(text)
        theme.set_tone(self.state_label, tone)
        self.stop_btn.setVisible(self.state == "running")
        self.results_btn.setVisible(self.state == "finished")
        self.retry_btn.setVisible(self.state in ("failed", "cancelled"))
