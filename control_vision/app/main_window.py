from __future__ import annotations

import logging
import json
import sqlite3
import csv
import shutil
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QIcon, QShortcut
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSpinBox, QStatusBar, QVBoxLayout, QWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QDoubleSpinBox, QInputDialog, QHeaderView, QListWidgetItem, QCheckBox, QTextBrowser,
    QGroupBox, QFrame, QGridLayout, QProgressBar, QScrollArea, QSplitter,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

from control_vision.pipeline.ingest import VideoError, probe_video
from control_vision.pipeline.tracking import track_video
from control_vision.pipeline.manual_tracking import track_seeded_video
from control_vision.pipeline.cycles import count_zone_cycles
from control_vision.pipeline.clips import create_highlight, thumbnail
from control_vision.pipeline.stats import robot_statistics
from control_vision.pipeline.live_score import propose_events, read_live_overlay
from control_vision.pipeline.scoreboard_ocr import read_scoreboard
from control_vision.scout.exports import export_scout_json, export_stats_csv, match_scoreboard
from control_vision.storage.database import ProjectDatabase
from control_vision.game.scoreboard import classify_match_time
from control_vision.app import theme

# Review and timeline rows keep the stored status and confidence for re-tinting.
EVENT_STATUS_ROLE = Qt.ItemDataRole.UserRole + 1
EVENT_CONFIDENCE_ROLE = Qt.ItemDataRole.UserRole + 2


class FrameCanvas(QLabel):
    clicked = Signal(int, int)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 225)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111827;color:#cbd5e1;border:1px solid #334155")
        self._frame_size = (0, 0)
        self.roi_points: list[list[float]] = []
        self.named_rois: dict[str, list[list[float]]] = {}

    def show_frame(self, frame: np.ndarray, annotations: list[tuple[int, int, str]] | None = None):
        self._last_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        image = QImage(rgb.data, w, h, channels * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        for index, (x, y, label) in enumerate(annotations or []):
            color = QColor("#fb7185" if index % 2 == 0 else "#60a5fa")
            painter.setPen(QPen(color, 4))
            painter.drawEllipse(x - 7, y - 7, 14, 14)
            painter.drawText(x + 10, y, label)
        roi_colors = {"goal_roi": "#facc15", "scoreboard_red_roi": "#fb7185", "scoreboard_blue_roi": "#60a5fa"}
        for name, points in self.named_rois.items():
            if len(points) < 3: continue
            painter.setPen(QPen(QColor(roi_colors.get(name, "#c4b5fd")), 2))
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        if len(self.roi_points) >= 3:
            painter.setPen(QPen(QColor("#facc15"), 3))
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in self.roi_points]))
        elif self.roi_points:
            painter.setPen(QPen(QColor("#facc15"), 3))
            for x, y in self.roi_points: painter.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
        painter.end()
        self._frame_size = (w, h)
        self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def mousePressEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None or not self._frame_size[0]:
            return
        x_pad = (self.width() - pixmap.width()) / 2
        y_pad = (self.height() - pixmap.height()) / 2
        x = (event.position().x() - x_pad) * self._frame_size[0] / pixmap.width()
        y = (event.position().y() - y_pad) * self._frame_size[1] / pixmap.height()
        if 0 <= x < self._frame_size[0] and 0 <= y < self._frame_size[1]:
            self.clicked.emit(round(x), round(y))

    def mouseDoubleClickEvent(self, event):
        self.finished.emit()


class FieldMap(QWidget):
    clicked = Signal(float, float)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(280, 280)
        self.tracks: dict[int, list[tuple[float, float, float]]] = {}
        self.current_time = 0.0
        self.zones: dict = {}
        self.draft: list[list[float]] = []

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(18, 18, -18, -18)
        painter.fillRect(rect, QColor("#14532d"))
        painter.setPen(QPen(QColor("#bbf7d0"), 2))
        painter.drawRect(rect)
        # The official field surface is six 24-inch tiles across, not a 3x3 grid.
        for i in range(1, 6):
            x, y = rect.left() + rect.width() * i / 6, rect.top() + rect.height() * i / 6
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
        for name, zone in self.zones.items():
            if zone.get("type", "").endswith("_roi"): continue
            points = zone.get("points", [])
            if len(points) < 3: continue
            polygon = QPolygonF([QPointF(rect.left() + x / 12 * rect.width(), rect.bottom() - y / 12 * rect.height()) for x, y in points])
            zone_color = QColor("#fde047") if zone.get("type") == "scoring" else QColor("#c4b5fd")
            painter.setPen(QPen(zone_color, 2)); painter.drawPolygon(polygon)
            painter.drawText(polygon.boundingRect().topLeft(), name)
        if len(self.draft) > 1:
            draft_polygon = QPolygonF([QPointF(rect.left() + x / 12 * rect.width(), rect.bottom() - y / 12 * rect.height()) for x, y in self.draft])
            painter.setPen(QPen(QColor("#facc15"), 2, Qt.PenStyle.DashLine)); painter.drawPolyline(draft_polygon)
        colors = [QColor("#fb7185"), QColor("#60a5fa"), QColor("#fda4af"), QColor("#93c5fd")]
        for index, points in self.tracks.items():
            visible = [(t_s, x, y) for t_s, x, y in points if self.current_time - 8 <= t_s <= self.current_time and 0 <= x <= 12 and 0 <= y <= 12]
            mapped = [(
                rect.left() + x / 12 * rect.width(),
                rect.bottom() - y / 12 * rect.height(),
            ) for _, x, y in visible]
            if len(mapped) < 1:
                continue
            painter.setPen(QPen(colors[index % len(colors)], 2))
            for first, second in zip(mapped, mapped[1:]):
                painter.drawLine(int(first[0]), int(first[1]), int(second[0]), int(second[1]))
            painter.setBrush(colors[index % len(colors)])
            painter.drawEllipse(int(mapped[-1][0]) - 5, int(mapped[-1][1]) - 5, 10, 10)

    def _field_rect(self):
        return self.rect().adjusted(18, 18, -18, -18)

    def mousePressEvent(self, event):
        rect = self._field_rect()
        pos = event.position()
        if rect.contains(int(pos.x()), int(pos.y())):
            x = (pos.x() - rect.left()) / rect.width() * 12
            y = (rect.bottom() - pos.y()) / rect.height() * 12
            self.clicked.emit(x, y)

    def mouseDoubleClickEvent(self, event):
        self.finished.emit()


class ProcessingWorker(QObject):
    progress = Signal(str, float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        started = time.monotonic()
        def report(stage, percent, detail):
            elapsed = time.monotonic() - started
            self.progress.emit(stage, percent, f"{detail} · elapsed {elapsed:.0f}s")
        try:
            if self.args[10] == "seeded_optical_flow":
                result = track_seeded_video(self.args[0], self.args[2], self.args[3], self.args[11],
                                            self.args[4], lambda: self.cancelled, report,
                                            self.args[5], self.args[6])
            else:
                result = track_video(*self.args[:5], cancel=lambda: self.cancelled,
                                     progress=report, logger=self.args[5], photos_dir=self.args[6], confidence_threshold=self.args[7], goal_roi=self.args[8], reid_model_path=self.args[9],
                                     alliances=self.args[11])
            self.done.emit((result, self.args[10]))
        except InterruptedError:
            self.failed.emit("Processing was cancelled.")
        except Exception as exc:
            self.args[5].exception("Processing failed")
            self.failed.emit(str(exc))


class ScoreboardOcrWorker(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, video_path: Path, red_roi: list[list[float]], blue_roi: list[list[float]]):
        super().__init__()
        self.video_path = video_path
        self.red_roi = red_roi
        self.blue_roi = blue_roi
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            self.done.emit(read_scoreboard(self.video_path, self.red_roi, self.blue_roi,
                                           progress=lambda percent, detail: self.progress.emit(percent, detail),
                                           cancel=lambda: self.cancelled))
        except InterruptedError:
            self.failed.emit("Scoreboard OCR was cancelled.")
        except Exception as exc:
            self.failed.emit(str(exc))


class LiveScoreWorker(QObject):
    """Reads the live score overlay and proposes events, off the UI thread."""
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, video_path: Path):
        super().__init__()
        self.video_path = video_path
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            reading = read_live_overlay(self.video_path,
                                        progress=lambda percent, detail: self.progress.emit(percent, detail),
                                        cancel=lambda: self.cancelled)
            proposal = propose_events(reading["samples"])
            self.done.emit({"complete": reading["complete"], "stopped_early": reading["stopped_early"],
                            "read_until_s": reading["read_until_s"], "duration_s": reading["duration_s"],
                            "events": proposal["events"], "match_start_s": proposal.get("match_start_s")})
        except InterruptedError:
            self.failed.emit("Reading the live score overlay was cancelled.")
        except Exception as exc:
            self.failed.emit(str(exc))


class IdentityDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robot identity")
        form = QFormLayout(self)
        self.team = QSpinBox(); self.team.setRange(1, 99999); self.team.setValue(1)
        self.alliance = QComboBox(); self.alliance.addItems(["red", "blue"])
        form.addRow("Team number", self.team)
        form.addRow("Alliance", self.alliance)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class PreviewDialog(QDialog):
    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Preview — {path.name}"); self.resize(800, 520)
        layout = QVBoxLayout(self); self.video = QVideoWidget(); layout.addWidget(self.video)
        controls = QHBoxLayout(); self.play = QPushButton("Play / Pause"); self.time = QLabel("Ready")
        controls.addWidget(self.play); controls.addWidget(self.time); layout.addLayout(controls)
        self.player = QMediaPlayer(self); self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio); self.player.setVideoOutput(self.video)
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.play.clicked.connect(lambda: self.player.play() if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState else self.player.pause())
        self.player.positionChanged.connect(lambda ms: self.time.setText(f"{ms / 1000:.1f}s"))
        self.finished.connect(lambda _: self.player.stop())


def missing_robot_labels(robots, robot_ids, track_rows) -> list[str]:
    """Labels of seeded robots that have no track rows, in seed order."""
    tracked = {int(row[0]) for row in track_rows}
    by_id = {int(robot["id"]): robot for robot in robots}
    labels = []
    for robot_id in robot_ids:
        if int(robot_id) in tracked:
            continue
        robot = by_id.get(int(robot_id))
        labels.append(f"team {robot['team_number']} ({robot['alliance']})" if robot else f"robot {robot_id}")
    return labels


def _video_switch_blocked(window) -> bool:
    if (getattr(window, "_processing_active", False) or getattr(window, "_ocr_active", False)
            or getattr(window, "_live_active", False)):
        QMessageBox.information(window, "Analysis in progress",
                                "Wait for processing or scoreboard OCR to finish, or cancel processing, before changing projects or videos.")
        return True
    for thread in (getattr(window, "_thread", None), getattr(window, "_ocr_thread", None),
                   getattr(window, "_live_thread", None)):
        try:
            if thread is not None and thread.isRunning():
                QMessageBox.information(window, "Analysis in progress",
                                        "Wait for processing or scoreboard OCR to finish, or cancel processing, before changing projects or videos.")
                return True
        except RuntimeError:
            # Qt may already have deleted a finished OCR thread.
            continue
    return False


def workflow_steps(*, project_open: bool, video_name: str | None, corners: int, field_zones: int, rois: int,
                   robots: int, analysis_status: str | None, processing: bool) -> list[dict]:
    """State of each setup step shown in the Analysis panel, in workflow order.

    Each step is ``{"key", "title", "state", "detail"}`` where state is one of
    done, current, todo, optional, running or failed. "current" marks the first
    required step that is not done; zones and ROIs are optional for processing.
    """
    video_done = video_name is not None
    steps = [
        {"key": "video", "title": "Match video", "done": video_done, "required": True,
         "detail": video_name if video_done else ("Import or load a match video." if project_open
                                                  else "Create or open a project first.")},
        {"key": "calibrate", "title": "Field calibration", "done": corners == 4, "required": True,
         "detail": f"{corners}/4 corners selected" if corners != 4 else "4/4 corners saved"},
        {"key": "zones", "title": "Zones and ROIs", "done": bool(field_zones or rois), "required": False,
         "detail": (f"{field_zones} zone(s) · {rois} ROI(s)" if field_zones or rois
                    else "Optional: draw zones for cycles and ROIs for scoreboard OCR.")},
        {"key": "robots", "title": "Robot identities", "done": robots > 0, "required": True,
         "detail": f"{robots}/4 robots seeded"},
        {"key": "process", "title": "Process", "done": analysis_status == "ready", "required": True,
         "detail": {"ready": "Analysis ready. Review flagged events next.",
                    "failed": "The last analysis did not finish.",
                    "running": "Analysis did not finish. Process again."}.get(
                        analysis_status or "", "Not processed yet.")},
    ]
    current_found = False
    for step in steps:
        if step["key"] == "process" and processing:
            step["state"], step["detail"] = "running", "Processing…"
            current_found = True
        elif step["key"] == "process" and analysis_status in {"failed", "running"} and video_done:
            step["state"] = "failed"
        elif step["done"]:
            step["state"] = "done"
        elif not step["required"]:
            step["state"] = "optional"
        elif not current_found and (project_open or step["key"] == "video"):
            step["state"] = "current"
            current_found = True
        else:
            step["state"] = "todo"
        del step["done"], step["required"]
    return steps


NAV_ENTRIES = (
    ("section", "Project"), (0, "Home"),
    ("section", "Match workflow"), (1, "Set up and process"), (3, "Review queue"), (4, "Scoring timeline"),
    (2, "Cycles and stats"), (5, "Highlights"), (6, "Score and export"),
    ("section", "App"), (7, "Settings"), (8, "BIOBUZZ rules"),
)


def _card() -> QFrame:
    frame = QFrame()
    frame.setProperty("card", True)
    return frame


def _label(text: str, role: str | None = None, tone: str | None = None, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if role: label.setProperty("role", role)
    if tone: label.setProperty("tone", tone)
    label.setWordWrap(wrap)
    return label


def _primary(button: QPushButton) -> QPushButton:
    button.setProperty("variant", "primary")
    return button


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Vision — Match analysis")
        self.resize(1360, 860)
        # Sidebar, setup steps and the minimum video and map widths fit in 1280 px.
        self.setMinimumSize(1280, 720)
        self.db: ProjectDatabase | None = None
        self.project_id = None
        self.video_id = None
        self.video_path: Path | None = None
        self.metadata = None
        self.corners: list[list[float]] = []
        self.zones: dict = {}
        self._drawing_zone = False
        self.seed_points: list[tuple[int, int, int]] = []
        self.robot_labels: list[str] = []
        self.tracks: dict[int, list[tuple[float, float, float]]] = {}
        self.cycles: list[dict] = []
        self.events = []
        self.photo_rows = []
        self.capture = None
        self.logger = None
        self.current_time = 0.0
        self.duration = 0.0
        self._project_name = None
        self._step_widgets: dict[str, tuple[QLabel, QLabel]] = {}
        self._theme_mode = theme.saved_mode()
        self._tokens = theme.apply_theme(QApplication.instance(), self._theme_mode)
        self.setStatusBar(QStatusBar())
        self.status_progress = QProgressBar(); self.status_progress.setRange(0, 100)
        self.status_progress.setFixedWidth(180); self.status_progress.setAccessibleName("Processing progress")
        self.status_progress.hide()
        self.context_label = QLabel("No project open")
        self.statusBar().addPermanentWidget(self.status_progress)
        self.statusBar().addPermanentWidget(self.context_label)

        # Analysis: setup steps on the left in workflow order, video and map on the right.
        root = QWidget(); root.setObjectName("Page"); layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(16)
        self.project_btn = _primary(QPushButton("&New project"))
        self.open_project_btn = QPushButton("&Open project")
        self.video_btn = QPushButton("Import video")
        self.load_video_btn = QPushButton("Load saved video")
        self.corner_btn = QPushButton("Select 4 corners")
        self.corner_btn.setToolTip("Click the field corners in order: top-left, top-right, bottom-right, bottom-left.")
        self.robot_btn = QPushButton("Seed a robot")
        self.robot_btn.setToolTip("Click a robot near its ground-contact point, then enter its team and alliance.")
        self.reuse_calibration_btn = QPushButton("Reuse calibration")
        self.reuse_calibration_btn.setToolTip("Copy corners and zones from another video in this project.")
        self.process_btn = _primary(QPushButton("Process match"))
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setProperty("variant", "danger")
        self.sample_rate = QComboBox(); self.sample_rate.addItems(["5", "6", "7", "8", "9", "10"]); self.sample_rate.setCurrentText("8")
        self.sample_rate.setAccessibleName("Sample rate (frames per second)")
        self.zone_type = QComboBox(); self.zone_type.addItems(["scoring", "pickup", "park"])
        self.zone_type.setAccessibleName("Zone type")
        self.zone_btn = QPushButton("Draw zone")
        self.goal_roi_btn = QPushButton("Draw scoring-element ROI")
        self.red_score_roi_btn = QPushButton("Red score ROI")
        self.red_score_roi_btn.setToolTip("Set red scoreboard ROI: outline the red score digits on the final results screen.")
        self.blue_score_roi_btn = QPushButton("Blue score ROI")
        self.blue_score_roi_btn.setToolTip("Set blue scoreboard ROI: outline the blue score digits on the final results screen.")
        self.zone_remove_combo = QComboBox(); self.zone_remove_combo.setAccessibleName("Zone or ROI to remove")
        self.zone_remove_btn = QPushButton("Remove")
        self.zone_remove_btn.setToolTip("Remove the selected zone or ROI")
        self.zone_help = _label("Draw zones on the map; double-click to finish a polygon. Geometry is user supplied.",
                                role="muted", wrap=True)
        self.zone_btn.setToolTip(self.zone_help.text())
        fps_label = QLabel("Sample &fps"); fps_label.setBuddy(self.sample_rate)

        steps_page = QWidget(); steps_layout = QVBoxLayout(steps_page)
        steps_layout.setContentsMargins(0, 0, 4, 0); steps_layout.setSpacing(8)
        steps_layout.addWidget(self._step_card("video", 1, "Match video", [[self.video_btn, self.load_video_btn]]))
        steps_layout.addWidget(self._step_card("calibrate", 2, "Field calibration", [[self.corner_btn, self.reuse_calibration_btn]]))
        steps_layout.addWidget(self._step_card("zones", 3, "Zones and ROIs", [
            [self.zone_type, self.zone_btn], [self.goal_roi_btn], [self.red_score_roi_btn, self.blue_score_roi_btn],
            [self.zone_remove_combo, self.zone_remove_btn]]))
        steps_layout.addWidget(self._step_card("robots", 4, "Robot identities", [[self.robot_btn]]))
        steps_layout.addWidget(self._step_card("process", 5, "Process", [[fps_label, self.sample_rate, self.cancel_btn], [self.process_btn]]))
        steps_layout.addStretch(1)
        steps_scroll = QScrollArea(); steps_scroll.setObjectName("PageScroll"); steps_scroll.setWidgetResizable(True)
        steps_scroll.setWidget(steps_page); steps_scroll.setFixedWidth(312)
        steps_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(steps_scroll)

        viewer = QVBoxLayout(); viewer.setSpacing(8)
        self.canvas = FrameCanvas(); self.field_map = FieldMap()
        self.canvas.setAccessibleName("Match video frame"); self.field_map.setAccessibleName("Field map")
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.addWidget(self.canvas); splitter.addWidget(self.field_map)
        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 2); splitter.setChildrenCollapsible(False)
        viewer.addWidget(splitter, 1)
        bottom = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, 1000)
        self.slider.setAccessibleName("Video position")
        self.time_label = QLabel("00:00 / 00:00"); self.time_label.setProperty("role", "muted")
        bottom.addWidget(self.slider, 1); bottom.addWidget(self.time_label)
        viewer.addLayout(bottom)
        self.progress_label = _label("Create a project, import a video, calibrate and seed robot identities to begin.",
                                     tone="info", wrap=True)
        self.progress_label.setAccessibleName("Analysis status")
        viewer.addWidget(self.progress_label)
        viewer.addWidget(self.zone_help)
        viewer.addWidget(_label("Run log", role="muted"))
        self.log_view = QListWidget(); self.log_view.setMaximumHeight(96); self.log_view.setAccessibleName("Run log")
        viewer.addWidget(self.log_view)
        layout.addLayout(viewer, 1)

        self.tabs = QTabWidget()
        home_page = QWidget(); home_page.setObjectName("Page"); home_layout = QHBoxLayout(home_page)
        home_layout.setContentsMargins(0, 0, 0, 0); home_layout.setSpacing(16)
        home_main = QVBoxLayout(); home_main.setSpacing(16)
        project_card = _card(); project_layout = QVBoxLayout(project_card); project_layout.setContentsMargins(20, 18, 20, 18)
        self.home_project_label = _label("No project is open", role="title")
        self.home_project_path = _label("Create a project folder, or open an existing Control Vision project database.",
                                        role="muted", wrap=True)
        self.home_project_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        project_actions = QHBoxLayout()
        project_actions.addWidget(self.project_btn); project_actions.addWidget(self.open_project_btn); project_actions.addStretch(1)
        project_layout.addWidget(self.home_project_label); project_layout.addWidget(self.home_project_path)
        project_layout.addSpacing(6); project_layout.addLayout(project_actions)
        home_main.addWidget(project_card)
        videos_card = _card(); videos_layout = QVBoxLayout(videos_card); videos_layout.setContentsMargins(20, 18, 20, 18)
        videos_layout.addWidget(_label("Match videos", role="section"))
        videos_layout.addWidget(_label("Recent videos in this project — double-click to load.", role="muted"))
        self.home_recent = QListWidget(); self.home_recent.setAccessibleName("Recent videos in this project")
        self.home_recent.itemDoubleClicked.connect(lambda item: self.load_saved_video_id(item.data(Qt.ItemDataRole.UserRole)))
        self.home_empty = _label("No videos yet. Import a match video to start.", role="muted")
        home_actions = QHBoxLayout()
        home_import = QPushButton("&Import video"); home_load = QPushButton("&Load selected video")
        home_import.clicked.connect(self.import_video)
        home_load.clicked.connect(lambda: self.load_saved_video_id(self.home_recent.currentItem().data(Qt.ItemDataRole.UserRole) if self.home_recent.currentItem() else None))
        home_actions.addWidget(home_import); home_actions.addWidget(home_load); home_actions.addStretch(1)
        videos_layout.addWidget(self.home_recent, 1); videos_layout.addWidget(self.home_empty); videos_layout.addLayout(home_actions)
        home_main.addWidget(videos_card, 1)
        home_layout.addLayout(home_main, 1)
        guide_card = _card(); guide_card.setFixedWidth(330); guide_layout = QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(20, 18, 20, 18)
        guide_layout.addWidget(_label("Workflow", role="section"))
        guide = _label(
            "<ol style='margin-left:-18px'>"
            "<li><b>Import</b> a match video into the project.</li>"
            "<li><b>Calibrate</b>: click the four field corners.</li>"
            "<li><b>Zones</b>: draw pickup, scoring and park zones and any ROIs.</li>"
            "<li><b>Seed robots</b>: click each robot and enter its team.</li>"
            "<li><b>Process</b> the match on this computer.</li>"
            "<li><b>Review</b> every flagged event against the video.</li>"
            "<li><b>Export</b> statistics, Scout JSON and clips.</li></ol>"
            "<p>Candidates stay <b>pending review</b> until you confirm them. Unknown beats wrong: "
            "leave a value unassigned when the video does not show it.</p>", wrap=True)
        guide.setTextFormat(Qt.TextFormat.RichText)
        guide_layout.addWidget(guide); guide_layout.addStretch(1)
        home_layout.addWidget(guide_card)
        self.tabs.addTab(home_page, "Home / Projects")
        self.tabs.addTab(root, "Analysis")
        self.cycles_table = QTableWidget(0, 6)
        self.cycles_table.setHorizontalHeaderLabels(["Team", "Alliance", "Start", "End", "Pickup", "Scoring"])
        self.cycles_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cycles_table.cellDoubleClicked.connect(self.jump_to_cycle)
        cycles_page = QWidget(); cycles_page.setObjectName("Page"); cycles_layout = QVBoxLayout(cycles_page)
        cycles_layout.setContentsMargins(0, 0, 0, 0)
        self.stats_table = QTableWidget(0, 9)
        self.stats_table.setHorizontalHeaderLabels(["Team", "Alliance", "Cycles", "Avg cycle (s)", "Top speed (ft/s)", "Distance (ft)", "Time per zone (s)", "Confirmed points", "Ranking points"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        cycles_layout.addWidget(_label("Zone-trip cycle list. Cycle counts use pickup-to-scoring zone transitions and have medium confidence until ball detection is available.", tone="info", wrap=True))
        cycles_layout.addWidget(_label("Zone-trip cycles — double-click to seek", role="section"))
        cycles_layout.addWidget(self.cycles_table, 3); cycles_layout.addWidget(_label("Per-robot movement statistics", role="section")); cycles_layout.addWidget(self.stats_table, 2)
        self.tabs.addTab(cycles_page, "Cycles && Stats")
        self.review_table = QTableWidget(0, 9)
        self.review_table.setHorizontalHeaderLabels(["Time", "Team", "Robot alliance", "Points to", "Event type", "Points", "Confidence", "Status", "Clip"])
        self.review_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.review_table.cellDoubleClicked.connect(self.preview_review_event)
        review_page = QWidget(); review_page.setObjectName("Page"); review_layout = QVBoxLayout(review_page)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.addWidget(_label("Zone crossings are candidates only. Enter points after reviewing the video clip; unconfirmed events do not count toward the video total.", tone="pending", wrap=True))
        review_layout.addWidget(_label("Pending events — double-click to preview the clip or seek the video", role="muted"))
        review_layout.addWidget(self.review_table, 1)
        scoring_config_path = Path(__file__).resolve().parents[1] / "config" / "scoring_biobuzz.json"
        self.scoring_config = json.loads(scoring_config_path.read_text(encoding="utf-8"))
        self.event_types = {entry["id"]: entry for entry in self.scoring_config["review_event_types"]}
        self.review_robot = QComboBox(); self.review_points = QSpinBox(); self.review_points.setRange(-999, 999)
        self.review_robot.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.review_robot.setMinimumContentsLength(16)
        self.review_alliance = QComboBox(); self.review_alliance.addItem("Robot alliance", None); self.review_alliance.addItem("Red", "red"); self.review_alliance.addItem("Blue", "blue")
        self.review_confidence = QComboBox(); self.review_confidence.addItems(["low", "medium", "high"])
        self.review_type = QComboBox(); self.review_type.addItem("Unclassified / other", "unclassified")
        for event_type in self.scoring_config["review_event_types"]: self.review_type.addItem(event_type["label"], event_type["id"])
        self.review_type.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.review_type.setMinimumContentsLength(16)
        self.confirm_btn = _primary(QPushButton("&Confirm")); self.reject_btn = QPushButton("Re&ject")
        self.reject_btn.setProperty("variant", "danger")
        review_box = QGroupBox("Review selected event"); review_grid = QGridLayout(review_box)
        review_grid.setHorizontalSpacing(12); review_grid.setVerticalSpacing(6)
        for column, (text, widget, name) in enumerate((
                ("&Event type", self.review_type, "Event type"),
                ("Attributed &robot", self.review_robot, "Attributed robot"),
                ("&Points", self.review_points, "Points"),
                ("Points &go to", self.review_alliance, "Alliance receiving the points"),
                ("Con&fidence", self.review_confidence, "Confidence"))):
            label = QLabel(text); label.setBuddy(widget); widget.setAccessibleName(name)
            review_grid.addWidget(label, 0, column); review_grid.addWidget(widget, 1, column)
        review_grid.setColumnStretch(0, 3); review_grid.setColumnStretch(1, 3)
        review_buttons = QHBoxLayout(); review_buttons.addWidget(self.confirm_btn); review_buttons.addWidget(self.reject_btn)
        review_grid.addLayout(review_buttons, 1, 5)
        self.review_rule_hint = _label("Scoring is awarded to an alliance under the manual; select robot attribution and point recipient separately.", tone="info", wrap=True)
        review_grid.addWidget(self.review_rule_hint, 2, 0, 1, 6)
        review_layout.addWidget(review_box)
        self.tabs.addTab(review_page, "Review Queue")
        self.scoring_table = QTableWidget(0, 8)
        self.scoring_table.setHorizontalHeaderLabels(["Time", "Team", "Robot alliance", "Points to", "Event type", "Points", "Confidence", "Status"])
        self.scoring_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scoring_table.cellDoubleClicked.connect(self.jump_to_event)
        scoring_page = QWidget(); scoring_page.setObjectName("Page"); scoring_layout = QVBoxLayout(scoring_page)
        scoring_layout.setContentsMargins(0, 0, 0, 0)
        self.add_event_btn = QPushButton("Add scoring event at current video time")
        scoring_actions = QHBoxLayout(); scoring_actions.addWidget(_label("All events — double-click to seek the video", role="muted"), 1)
        scoring_actions.addWidget(self.add_event_btn)
        scoring_layout.addLayout(scoring_actions); scoring_layout.addWidget(self.scoring_table, 1)
        self.tabs.addTab(scoring_page, "Scoring Timeline")
        self.clip_list = QListWidget(); self.clip_list.itemDoubleClicked.connect(self.preview_selected_clip)
        self.clip_list.setIconSize(QSize(120, 68)); self.clip_list.setSpacing(2); self.clip_list.setAccessibleName("Highlight clips")
        clips_page = QWidget(); clips_page.setObjectName("Page"); clips_layout = QVBoxLayout(clips_page)
        clips_layout.setContentsMargins(0, 0, 0, 0)
        self.generate_clips_btn = _primary(QPushButton("Generate highlight clips for scoring candidates"))
        self.preview_clip_btn = QPushButton("Preview selected clip")
        self.clip_export_btn = QPushButton("Export selected clip")
        clips_controls = QHBoxLayout(); clips_controls.addWidget(self.generate_clips_btn); clips_controls.addWidget(self.preview_clip_btn); clips_controls.addWidget(self.clip_export_btn); clips_controls.addStretch(1)
        clips_layout.addLayout(clips_controls); clips_layout.addWidget(self.clip_list, 1); self.tabs.addTab(clips_page, "Highlights")
        export_page = QWidget(); export_page.setObjectName("Page"); export_layout = QVBoxLayout(export_page)
        export_layout.setContentsMargins(0, 0, 8, 0); export_layout.setSpacing(4)
        self.official_red = QLineEdit(); self.official_blue = QLineEdit(); self.official_red.setPlaceholderText("Optional"); self.official_blue.setPlaceholderText("Optional")
        self.official_red.setAccessibleName("Scoreboard / official red total"); self.official_blue.setAccessibleName("Scoreboard / official blue total")
        score_box = QGroupBox("Official score"); score_layout = QVBoxLayout(score_box)
        official_row = QHBoxLayout(); official_row.addWidget(QLabel("Scoreboard / official red total")); official_row.addWidget(self.official_red); official_row.addSpacing(16); official_row.addWidget(QLabel("Scoreboard / official blue total")); official_row.addWidget(self.official_blue)
        score_layout.addLayout(official_row)
        self.read_scoreboard_btn = QPushButton("Read final scoreboard from video")
        self.use_ocr_btn = QPushButton("Use OCR reading in score comparison")
        self.use_ocr_btn.setEnabled(False)
        self.scoreboard_ocr_status = _label("Seek to the final results screen, then draw tight red and blue score ROIs in Analysis. OCR proposes a multi-frame reading for your review.", tone="info", wrap=True)
        self.scoreboard_ocr_status.setAccessibleName("Scoreboard OCR status")
        ocr_row = QHBoxLayout(); ocr_row.addWidget(self.read_scoreboard_btn); ocr_row.addWidget(self.use_ocr_btn); ocr_row.addStretch(1)
        score_layout.addLayout(ocr_row); score_layout.addWidget(self.scoreboard_ocr_status)
        self.read_live_score_btn = QPushButton("Read live score overlay")
        self.read_live_score_btn.setToolTip("Read the FTC Live overlay through the match and add its HIVE TIPS, foul credits "
                                            "and other score changes to Review as flagged events.")
        self.live_score_status = _label("The live overlay gives each alliance's score changes with their time. Events are "
                                        "added to Review flagged, with no robot: choose the robot only if the video shows it.",
                                        tone="info", wrap=True)
        self.live_score_status.setAccessibleName("Live score overlay status")
        live_row = QHBoxLayout(); live_row.addWidget(self.read_live_score_btn); live_row.addStretch(1)
        score_layout.addLayout(live_row); score_layout.addWidget(self.live_score_status)
        self.score_compare = _label("Video totals include confirmed events only.", tone="info", wrap=True)
        self.score_compare.setAccessibleName("Score comparison")
        score_layout.addWidget(self.score_compare)
        export_layout.addWidget(score_box)
        export_layout.addWidget(self._build_ruling_box())
        rp_box = QGroupBox("Ranking Point thresholds"); rp_layout = QVBoxLayout(rp_box)
        rp_row = QHBoxLayout()
        self.swarm_threshold = QSpinBox(); self.swarm_threshold.setRange(0, 999); self.swarm_threshold.setValue(16)
        self.pollinator_1_threshold = QSpinBox(); self.pollinator_1_threshold.setRange(0, 99); self.pollinator_1_threshold.setValue(4)
        self.pollinator_2_threshold = QSpinBox(); self.pollinator_2_threshold.setRange(0, 99); self.pollinator_2_threshold.setValue(7)
        for text, spin in (("SWARM threshold", self.swarm_threshold), ("POLLINATOR 1 tips", self.pollinator_1_threshold), ("POLLINATOR 2 tips", self.pollinator_2_threshold)):
            spin.setAccessibleName(text); rp_row.addWidget(QLabel(text)); rp_row.addWidget(spin); rp_row.addSpacing(12)
        rp_row.addStretch(1)
        rp_layout.addLayout(rp_row)
        self.ranking_summary = _label("Ranking Points follow the selected event thresholds.", wrap=True); rp_layout.addWidget(self.ranking_summary)
        export_layout.addWidget(rp_box)
        self.csv_export_btn = QPushButton("Export statistics CSV"); self.json_export_btn = _primary(QPushButton("Export Scout JSON"))
        self.photos_export_btn = QPushButton("Export robot photos")
        files_box = QGroupBox("Export files"); export_row = QHBoxLayout(files_box)
        export_row.addWidget(self.json_export_btn); export_row.addWidget(self.csv_export_btn); export_row.addWidget(self.photos_export_btn); export_row.addStretch(1)
        export_layout.addWidget(files_box); export_layout.addStretch(1)
        self.tabs.addTab(self._scroll_page(export_page), "Export")
        settings_page = QWidget(); settings_page.setObjectName("Page"); settings_outer = QVBoxLayout(settings_page)
        settings_outer.setContentsMargins(0, 0, 8, 0)
        tracking_box = QGroupBox("Tracking"); settings_layout = QFormLayout(tracking_box)
        settings_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft); settings_layout.setVerticalSpacing(10)
        self.model_path_edit = QLineEdit(); self.reid_path_edit = QLineEdit(); self.ffmpeg_path_edit = QLineEdit()
        self.tracking_mode = QComboBox(); self.tracking_mode.addItem("YOLO11 local model (recommended with trained weights)", "yolo")
        self.tracking_mode.addItem("Seeded optical flow preview (no model; not accurate enough for scoring)", "seeded_optical_flow")
        self.model_browse_btn = QPushButton("Browse…"); self.reid_browse_btn = QPushButton("Browse…"); self.ffmpeg_browse_btn = QPushButton("Browse…")
        model_row = QHBoxLayout(); model_row.addWidget(self.model_path_edit); model_row.addWidget(self.model_browse_btn)
        reid_row = QHBoxLayout(); reid_row.addWidget(self.reid_path_edit); reid_row.addWidget(self.reid_browse_btn)
        ffmpeg_row = QHBoxLayout(); ffmpeg_row.addWidget(self.ffmpeg_path_edit); ffmpeg_row.addWidget(self.ffmpeg_browse_btn)
        self.confidence_spin = QDoubleSpinBox(); self.confidence_spin.setRange(0.05, 0.95); self.confidence_spin.setSingleStep(0.05); self.confidence_spin.setValue(0.25)
        self.settings_save_btn = _primary(QPushButton("Save settings"))
        settings_layout.addRow("Tracking method", self.tracking_mode); settings_layout.addRow("YOLO11n ONNX model", model_row); settings_layout.addRow("Optional local ReID embeddings (ONNX) or compatible YOLO .pt", reid_row); settings_layout.addRow("FFmpeg executable (blank uses PATH)", ffmpeg_row); settings_layout.addRow("Detection confidence", self.confidence_spin)
        settings_layout.addRow(_label("Tracking settings and the sample rate (set in Set up and process) are saved with the active project.", role="muted", wrap=True))
        save_row = QHBoxLayout(); save_row.addWidget(self.settings_save_btn); save_row.addStretch(1)
        settings_layout.addRow(save_row)
        settings_outer.addWidget(tracking_box)
        appearance_box = QGroupBox("Appearance"); appearance_layout = QFormLayout(appearance_box)
        self.theme_combo = QComboBox()
        for label_text, mode in (("Match Windows", "system"), ("Light", "light"), ("Dark", "dark")):
            self.theme_combo.addItem(label_text, mode)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(self._theme_mode)))
        self.theme_combo.setAccessibleName("Theme")
        theme_label = QLabel("&Theme"); theme_label.setBuddy(self.theme_combo)
        appearance_layout.addRow(theme_label, self.theme_combo)
        appearance_layout.addRow(_label("Saved for this computer, not with the project.", role="muted"))
        settings_outer.addWidget(appearance_box); settings_outer.addStretch(1)
        self.tabs.addTab(self._scroll_page(settings_page), "Settings")
        rules_page = QWidget(); rules_page.setObjectName("Page"); rules_layout = QVBoxLayout(rules_page)
        rules_layout.setContentsMargins(0, 0, 0, 0)
        self.rules_browser = QTextBrowser(); self.rules_browser.setOpenExternalLinks(True)
        self.rules_browser.setMarkdown(self.build_rules_reference())
        rules_layout.addWidget(self.rules_browser); self.tabs.addTab(rules_page, "BIOBUZZ Rules")
        for table in (self.cycles_table, self.stats_table, self.review_table, self.scoring_table):
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setShowGrid(False)
            table.verticalHeader().setVisible(False)
            table.verticalHeader().setDefaultSectionSize(30)
        for table in (self.review_table, self.scoring_table):
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for table, name in ((self.cycles_table, "Zone-trip cycles"), (self.stats_table, "Per-robot statistics"),
                            (self.review_table, "Review queue"), (self.scoring_table, "Scoring timeline")):
            table.setAccessibleName(name)
        self.tabs.tabBar().hide()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self._build_shell())
        self.tabs.currentChanged.connect(self._tab_changed)

        self.project_btn.clicked.connect(self.new_project)
        self.open_project_btn.clicked.connect(self.open_project)
        self.video_btn.clicked.connect(self.import_video)
        self.load_video_btn.clicked.connect(self.load_saved_video)
        self.corner_btn.clicked.connect(self.select_corner)
        self.robot_btn.clicked.connect(self.seed_robot)
        self.zone_btn.clicked.connect(self.start_zone_drawing)
        self.goal_roi_btn.clicked.connect(self.start_roi_drawing)
        self.red_score_roi_btn.clicked.connect(lambda: self.start_named_roi_drawing("scoreboard_red_roi"))
        self.blue_score_roi_btn.clicked.connect(lambda: self.start_named_roi_drawing("scoreboard_blue_roi"))
        self.zone_remove_btn.clicked.connect(self.remove_selected_zone)
        self.reuse_calibration_btn.clicked.connect(self.reuse_calibration)
        self.process_btn.clicked.connect(self.process)
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.canvas.clicked.connect(self.canvas_click)
        self.canvas.finished.connect(self.finish_roi)
        self.field_map.clicked.connect(self.zone_map_click)
        self.field_map.finished.connect(self.finish_zone)
        self.slider.valueChanged.connect(self.seek)
        self.confirm_btn.clicked.connect(lambda: self.review_selected("confirmed"))
        self.reject_btn.clicked.connect(lambda: self.review_selected("rejected"))
        self.review_type.currentIndexChanged.connect(self.review_type_changed)
        self.review_robot.currentIndexChanged.connect(self.review_type_changed)
        self.review_table.currentCellChanged.connect(self.load_review_selection)
        self.add_event_btn.clicked.connect(self.add_manual_event)
        self.generate_clips_btn.clicked.connect(self.generate_clips)
        self.preview_clip_btn.clicked.connect(self.preview_selected_clip)
        self.clip_export_btn.clicked.connect(self.export_selected_clip)
        self.csv_export_btn.clicked.connect(self.export_csv)
        self.json_export_btn.clicked.connect(self.export_json)
        self.photos_export_btn.clicked.connect(self.export_photos)
        self.read_scoreboard_btn.clicked.connect(self.start_scoreboard_ocr)
        self.read_live_score_btn.clicked.connect(self.start_live_score)
        self.use_ocr_btn.clicked.connect(self.apply_scoreboard_ocr)
        self.official_red.textChanged.connect(self.update_score_comparison)
        self.official_blue.textChanged.connect(self.update_score_comparison)
        self.review_alliance.currentIndexChanged.connect(self.update_score_comparison)
        self.official_red.editingFinished.connect(self.save_official_scores)
        self.official_blue.editingFinished.connect(self.save_official_scores)
        self.match_type_combo.currentIndexChanged.connect(self.match_type_changed)
        for control in (self.disq_red, self.disq_blue, self.match_final_check):
            control.toggled.connect(self.save_match_ruling)
        self.ruling_combo.currentIndexChanged.connect(self.save_match_ruling)
        self.swarm_threshold.editingFinished.connect(self.save_official_scores); self.pollinator_1_threshold.editingFinished.connect(self.save_official_scores); self.pollinator_2_threshold.editingFinished.connect(self.save_official_scores)
        self.swarm_threshold.valueChanged.connect(self.update_score_comparison); self.pollinator_1_threshold.valueChanged.connect(self.update_score_comparison); self.pollinator_2_threshold.valueChanged.connect(self.update_score_comparison)
        self.settings_save_btn.clicked.connect(self.save_settings)
        self.model_browse_btn.clicked.connect(self.browse_model)
        self.reid_browse_btn.clicked.connect(self.browse_reid)
        self.ffmpeg_browse_btn.clicked.connect(self.browse_ffmpeg)
        self.theme_combo.currentIndexChanged.connect(self.theme_changed)
        try:
            QApplication.instance().styleHints().colorSchemeChanged.connect(self._system_scheme_changed)
        except AttributeError:  # Qt older than 6.5 has no colour-scheme hint.
            pass
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.refresh_video)
        self._mode = None
        self._worker = None
        self._thread = None
        self._processing_active = False
        self._ocr_worker = None
        self._ocr_thread = None
        self._ocr_active = False
        self._live_worker = None
        self._live_thread = None
        self._live_active = False
        self._live_video_id = None
        self.scoreboard_ocr = None
        # No video is loaded yet, so there are no results to show.
        self._set_results_enabled(False)
        self._tab_changed(self.tabs.currentIndex())
        self._refresh_workflow()

    # ----- Shell: sidebar navigation, page header, theme -----

    def _build_shell(self) -> QWidget:
        shell = QWidget(); shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0); shell_layout.setSpacing(0)
        sidebar = QFrame(); sidebar.setObjectName("Sidebar"); sidebar.setFixedWidth(212)
        side_layout = QVBoxLayout(sidebar); side_layout.setContentsMargins(12, 16, 12, 12); side_layout.setSpacing(4)
        app_name = QLabel("Control Vision"); app_name.setObjectName("AppName")
        tagline = QLabel("BIOBUZZ match scouting"); tagline.setObjectName("AppTagline")
        side_layout.addWidget(app_name); side_layout.addWidget(tagline); side_layout.addSpacing(12)
        self.nav = QListWidget(); self.nav.setObjectName("Nav"); self.nav.setAccessibleName("Sections")
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav_rows: dict[int, int] = {}
        self._nav_labels: dict[int, str] = {}
        self._nav_tooltips: dict[int, str] = {}
        self._review_pending = 0
        section_font = QFont(self.font()); section_font.setPointSizeF(max(7.5, section_font.pointSizeF() - 1)); section_font.setBold(True)
        shortcut_number = 0
        for key, text in NAV_ENTRIES:
            item = QListWidgetItem(text.upper() if key == "section" else text)
            if key == "section":
                item.setFlags(Qt.ItemFlag.NoItemFlags); item.setFont(section_font)
                item.setSizeHint(QSize(0, 30))
            else:
                item.setData(Qt.ItemDataRole.UserRole, key)
                shortcut_number += 1
                self._nav_tooltips[key] = f"{text} (Ctrl+{shortcut_number})"
                item.setToolTip(self._nav_tooltips[key])
                shortcut = QShortcut(QKeySequence(f"Ctrl+{shortcut_number}"), self)
                shortcut.activated.connect(lambda index=key: self._go_to_tab(index))
                self._nav_rows[key] = self.nav.count()
                self._nav_labels[key] = text
            self.nav.addItem(item)
        self.nav.currentItemChanged.connect(self._nav_selected)
        side_layout.addWidget(self.nav, 1)
        shell_layout.addWidget(sidebar)
        content = QWidget(); content.setObjectName("Page"); content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 18, 24, 16); content_layout.setSpacing(12)
        self.page_title = QLabel(""); self.page_title.setObjectName("PageTitle")
        self.page_context = QLabel(""); self.page_context.setObjectName("PageContext")
        header = QVBoxLayout(); header.setSpacing(2); header.addWidget(self.page_title); header.addWidget(self.page_context)
        content_layout.addLayout(header)
        content_layout.addWidget(self.tabs, 1)
        shell_layout.addWidget(content, 1)
        return shell

    def _scroll_page(self, page: QWidget) -> QScrollArea:
        scroll = QScrollArea(); scroll.setObjectName("PageScroll"); scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    def _step_card(self, key: str, number: int, title: str, rows) -> QFrame:
        card = _card(); card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 8, 12, 10); card_layout.setSpacing(6)
        head = QHBoxLayout(); head.setSpacing(8)
        number_label = _label(str(number), role="step-number")
        head.addWidget(number_label); head.addWidget(_label(title, role="section"), 1)
        card_layout.addLayout(head)
        state_label = _label("", role="step-state", wrap=True)
        state_label.setAccessibleName(f"{title} status")
        card_layout.addWidget(state_label)
        for row in rows:
            if isinstance(row, QWidget):
                card_layout.addWidget(row); continue
            row_layout = QHBoxLayout(); row_layout.setSpacing(6)
            for widget in row: row_layout.addWidget(widget, 0 if isinstance(widget, QLabel) else 1)
            card_layout.addLayout(row_layout)
        self._step_widgets[key] = (number_label, state_label)
        return card

    def _go_to_tab(self, index: int):
        if self.tabs.isTabEnabled(index):
            self.tabs.setCurrentIndex(index)

    def _nav_selected(self, current, previous):
        if current is None: return
        index = current.data(Qt.ItemDataRole.UserRole)
        if index is None: return
        if self.tabs.isTabEnabled(int(index)):
            self.tabs.setCurrentIndex(int(index))
        else:
            self._sync_nav()

    def _tab_changed(self, index: int):
        row = self._nav_rows.get(index)
        if row is not None and self.nav.currentRow() != row:
            self.nav.blockSignals(True); self.nav.setCurrentRow(row); self.nav.blockSignals(False)
        self.page_title.setText(self._nav_labels.get(index, self.tabs.tabText(index)))

    def _sync_nav(self):
        """Mirror tab availability and the current tab in the sidebar."""
        for index, row in self._nav_rows.items():
            item = self.nav.item(row)
            flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled if self.tabs.isTabEnabled(index) else Qt.ItemFlag.NoItemFlags
            item.setFlags(flags)
            item.setToolTip(self._nav_tooltip(index))
        self._tab_changed(self.tabs.currentIndex())

    def _nav_tooltip(self, index: int) -> str:
        if index in (2, 3, 4, 5, 6) and not self.tabs.isTabEnabled(index):
            return f"{self._nav_labels[index]}: process the video to see results"
        if index == 3 and self._review_pending:
            return f"Review queue: {self._review_pending} event(s) pending review (Ctrl+3)"
        return self._nav_tooltips[index]

    def _set_review_badge(self, flagged: int):
        item = self.nav.item(self._nav_rows[3])
        item.setText(f"Review queue  ({flagged})" if flagged else "Review queue")
        self._review_pending = flagged
        item.setToolTip(self._nav_tooltip(3))
        item.setForeground(QColor(self._tokens.pending_fg) if flagged else QColor(self._tokens.text))

    def theme_changed(self, *_):
        mode = self.theme_combo.currentData()
        if mode not in theme.THEME_MODES: return
        theme.save_mode(mode)
        self._theme_mode = mode
        self._apply_theme()

    def _system_scheme_changed(self, *_):
        if self._theme_mode == "system":
            self._apply_theme()

    def _apply_theme(self):
        self._tokens = theme.apply_theme(QApplication.instance(), self._theme_mode)
        for table in (self.review_table, self.scoring_table):
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item is not None:
                    self._paint_event_row(table, row, item.data(EVENT_STATUS_ROLE), item.data(EVENT_CONFIDENCE_ROLE))
        flagged = sum(1 for event in self.events if event["status"] == "flagged")
        self._set_review_badge(flagged)

    def _paint_event_row(self, table: QTableWidget, row: int, status, confidence):
        """Tint a review or timeline row so pending and low-confidence events stand out."""
        first = table.item(row, 0)
        if first is not None:
            first.setData(EVENT_STATUS_ROLE, status); first.setData(EVENT_CONFIDENCE_ROLE, confidence)
        background, foreground = theme.row_colours(self._tokens, status, confidence)
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is None: continue
            item.setBackground(QBrush(QColor(background)) if background else QBrush())
            item.setForeground(QBrush(QColor(foreground)) if foreground else QBrush())

    def _show_status(self, text: str, tone: str = "info"):
        """Show an analysis message; every message sets its own tone so none inherits the last one."""
        theme.set_tone(self.progress_label, tone)
        self.progress_label.setText(text)

    def _refresh_workflow(self):
        """Update the setup steps, the page context line and the status bar context."""
        status = None
        if self.db and self.video_id is not None:
            # This only updates labels and runs on error paths too (a failed
            # save, a locked database), so it must never raise over them.
            try:
                row = self.db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (self.video_id,)).fetchone()
                status = str(row["analysis_status"]) if row else None
            except (sqlite3.Error, TypeError, KeyError, IndexError):
                status = None
        field_zones = sum(1 for name in self.zones if not str(name).endswith("_roi"))
        steps = workflow_steps(project_open=bool(self.db and self.project_id is not None),
                               video_name=self.video_path.name if self.video_path else None,
                               corners=len(self.corners), field_zones=field_zones, rois=len(self.zones) - field_zones,
                               robots=len(self.seed_points), analysis_status=status,
                               processing=self._processing_active)
        for step in steps:
            number_label, state_label = self._step_widgets[step["key"]]
            theme.set_state(number_label, step["state"]); theme.set_state(state_label, step["state"])
            number_label.setText("✓" if step["state"] == "done" else str(list(self._step_widgets).index(step["key"]) + 1))
            state_label.setText(step["detail"])
        parts = [self._project_name or "No project open"]
        if self.video_path is not None:
            parts.append(self.video_path.name)
        self.page_context.setText("  ·  ".join(parts))
        self.context_label.setText("  ·  ".join(parts))

    def project_dir(self) -> Path | None:
        return self.db.path.parent if self.db else None

    def _build_ruling_box(self) -> QGroupBox:
        """Match type and the reviewer-entered referee ruling (T601, 13.7, T705)."""
        self._loading_ruling = False
        self.team_dq_checks: dict[int, QCheckBox] = {}
        box = QGroupBox("Match type and referee ruling")
        layout = QVBoxLayout(box)
        type_row = QHBoxLayout()
        self.match_type_combo = QComboBox()
        self.match_type_combo.addItem("Qualification", "qualification")
        self.match_type_combo.addItem("Playoff", "playoff")
        self.match_type_combo.setAccessibleName("Match type")
        type_label = QLabel("&Match type"); type_label.setBuddy(self.match_type_combo)
        type_row.addWidget(type_label); type_row.addWidget(self.match_type_combo); type_row.addStretch(1)
        layout.addLayout(type_row)

        self.qualification_panel = QWidget()
        qualification_layout = QVBoxLayout(self.qualification_panel); qualification_layout.setContentsMargins(0, 0, 0, 0)
        qualification_note = QLabel("Qualification DQ applies only to the disqualified team: it gets 0 match points and "
                                    "0 Ranking Points; its alliance partner is unaffected (T601). Enter only the Head Referee's ruling.")
        qualification_note.setWordWrap(True)
        qualification_layout.addWidget(qualification_note)
        self.team_dq_row = QHBoxLayout()
        self.team_dq_empty = QLabel("Seed robots to record a team DQ.")
        self.team_dq_row.addWidget(self.team_dq_empty); self.team_dq_row.addStretch(1)
        qualification_layout.addLayout(self.team_dq_row)
        layout.addWidget(self.qualification_panel)

        self.playoff_panel = QWidget()
        playoff_layout = QVBoxLayout(self.playoff_panel); playoff_layout.setContentsMargins(0, 0, 0, 0)
        playoff_note = QLabel("Playoff DQ applies to the whole alliance, which gets 0 match points. Playoffs award no "
                              "Ranking Points (13.7). If both alliances are DQ'd, enter the Head Referee's ruling (T705).")
        playoff_note.setWordWrap(True)
        playoff_layout.addWidget(playoff_note)
        alliance_row = QHBoxLayout()
        self.disq_red = QCheckBox("&Red alliance DQ"); self.disq_blue = QCheckBox("&Blue alliance DQ")
        self.disq_red.setAccessibleName("Red alliance disqualified (playoff)")
        self.disq_blue.setAccessibleName("Blue alliance disqualified (playoff)")
        self.match_final_check = QCheckBox("Match &final (all evidence reviewed)")
        self.match_final_check.setAccessibleName("Playoff match final")
        self.match_final_check.setToolTip("Mark the playoff match final once every event is reviewed so the reviewed score decides the winner or tie.")
        for widget in (self.disq_red, self.disq_blue, self.match_final_check): alliance_row.addWidget(widget)
        alliance_row.addStretch(1)
        playoff_layout.addLayout(alliance_row)
        ruling_row = QHBoxLayout()
        self.ruling_combo = QComboBox()
        self.ruling_combo.addItem("Pending referee ruling", None)
        self.ruling_combo.addItem("Red disqualified first (blue wins)", "red")
        self.ruling_combo.addItem("Blue disqualified first (red wins)", "blue")
        self.ruling_combo.addItem("Simultaneous DQ (tie; match replayed)", "simultaneous")
        self.ruling_combo.setAccessibleName("Referee ruling for a double disqualification")
        self.ruling_label = QLabel("Referee &decision for double DQ (T705)"); self.ruling_label.setBuddy(self.ruling_combo)
        ruling_row.addWidget(self.ruling_label); ruling_row.addWidget(self.ruling_combo); ruling_row.addStretch(1)
        playoff_layout.addLayout(ruling_row)
        layout.addWidget(self.playoff_panel)

        self.ruling_status = QLabel("")
        self.ruling_status.setWordWrap(True)
        self.ruling_status.setAccessibleName("Match ruling status")
        layout.addWidget(self.ruling_status)
        self._sync_ruling_controls()
        return box

    def _selected_match_type(self) -> str:
        return str(self.match_type_combo.currentData() or "qualification")

    def _sync_ruling_controls(self):
        """Show only the controls that apply to the selected match type."""
        playoff = self._selected_match_type() == "playoff"
        self.qualification_panel.setVisible(not playoff)
        self.playoff_panel.setVisible(playoff)
        double_dq = self.disq_red.isChecked() and self.disq_blue.isChecked()
        self.ruling_combo.setEnabled(playoff and double_dq)
        self.ruling_label.setEnabled(playoff and double_dq)

    def _rebuild_team_dq_checks(self, robots, disqualified_ids: set[int]):
        for checkbox in self.team_dq_checks.values():
            self.team_dq_row.removeWidget(checkbox)
            checkbox.deleteLater()
        self.team_dq_checks = {}
        for index, robot in enumerate(robots):
            robot_id = int(robot["id"])
            checkbox = QCheckBox(f"Team {robot['team_number']} ({robot['alliance']}) DQ")
            checkbox.setAccessibleName(f"Team {robot['team_number']} {robot['alliance']} disqualified (qualification)")
            checkbox.setChecked(robot_id in disqualified_ids)
            checkbox.toggled.connect(self.save_match_ruling)
            self.team_dq_row.insertWidget(index, checkbox)
            self.team_dq_checks[robot_id] = checkbox
        self.team_dq_empty.setVisible(not robots)

    def load_match_ruling(self):
        """Show the stored match type and ruling; without a video, reset to a qualification match."""
        self._loading_ruling = True
        try:
            ruling = None
            robots = []
            if self.db and self.video_id is not None:
                ruling = self.db.match_ruling(self.video_id)
                robots = self.db.robots_for_video(self.video_id)
            match_type = ruling["match_type"] if ruling else "qualification"
            self.match_type_combo.setCurrentIndex(max(0, self.match_type_combo.findData(match_type)))
            alliances = ruling["playoff_disqualified"] if ruling else set()
            self.disq_red.setChecked("red" in alliances); self.disq_blue.setChecked("blue" in alliances)
            self.match_final_check.setChecked(bool(ruling and ruling["match_final"]))
            choice = None
            if ruling and ruling["simultaneous_disqualification"]:
                choice = "simultaneous"
            elif ruling:
                choice = ruling["first_disqualified"]
            self.ruling_combo.setCurrentIndex(max(0, self.ruling_combo.findData(choice)))
            self._rebuild_team_dq_checks(robots, ruling["qualification_disqualified_robot_ids"] if ruling else set())
        finally:
            self._loading_ruling = False
        self._sync_ruling_controls()

    def _ruling_from_controls(self) -> dict:
        if self._selected_match_type() == "qualification":
            return {"qualification_disqualified_robot_ids": sorted(
                robot_id for robot_id, checkbox in self.team_dq_checks.items() if checkbox.isChecked())}
        alliances = {alliance for alliance, checkbox in (("red", self.disq_red), ("blue", self.disq_blue))
                     if checkbox.isChecked()}
        choice = self.ruling_combo.currentData() if alliances == {"red", "blue"} else None
        return {"playoff_disqualified": alliances,
                "first_disqualified": choice if choice in {"red", "blue"} else None,
                "simultaneous_disqualification": choice == "simultaneous",
                "match_final": self.match_final_check.isChecked()}

    def save_match_ruling(self, *_):
        """Save the ruling shown in the controls; an invalid ruling is refused and the stored one shown."""
        if self._loading_ruling:
            return
        if not (self.disq_red.isChecked() and self.disq_blue.isChecked()) and self.ruling_combo.currentIndex() != 0:
            # A first-DQ or simultaneous ruling only exists for a double DQ.
            self._loading_ruling = True
            self.ruling_combo.setCurrentIndex(0)
            self._loading_ruling = False
        self._sync_ruling_controls()
        if not self.db or self.video_id is None:
            return
        try:
            self.db.set_match_ruling(self.video_id, self._selected_match_type(), **self._ruling_from_controls())
        except ValueError as exc:
            QMessageBox.warning(self, "Ruling not saved", str(exc))
            self.load_match_ruling()
        self.update_score_comparison()

    def match_type_changed(self, *_):
        if self._loading_ruling:
            return
        new_type = self._selected_match_type()
        if not self.db or self.video_id is None:
            self._sync_ruling_controls(); return
        ruling = self.db.match_ruling(self.video_id)
        if ruling["match_type"] == new_type:
            self._sync_ruling_controls(); return
        cleared = []
        if new_type == "playoff" and ruling["qualification_disqualified_robot_ids"]:
            teams = {int(robot["id"]): robot["team_number"] for robot in self.db.robots_for_video(self.video_id)}
            cleared.append("qualification team DQ for team " + ", ".join(
                str(teams[robot_id]) for robot_id in sorted(ruling["qualification_disqualified_robot_ids"])))
        if new_type == "qualification":
            if ruling["playoff_disqualified"]:
                cleared.append("playoff alliance DQ (" + ", ".join(sorted(ruling["playoff_disqualified"])) + ")")
            if ruling["first_disqualified"] or ruling["simultaneous_disqualification"]:
                cleared.append("double-DQ referee ruling")
            if ruling["match_final"]:
                cleared.append("playoff match final mark")
        if cleared:
            answer = QMessageBox.question(
                self, "Change match type",
                f"A {new_type} match cannot keep the " + "; ".join(cleared) +
                ". Change the match type and clear it?")
            if answer != QMessageBox.StandardButton.Yes:
                self.load_match_ruling()
                return
        try:
            self.db.set_match_type(self.video_id, new_type, clear_inapplicable=bool(cleared))
        except ValueError as exc:
            QMessageBox.warning(self, "Match type not changed", str(exc))
        self.load_match_ruling()
        self.update_score_comparison()

    def current_scoreboard(self, events, robots) -> dict:
        """The match board from the stored ruling and the thresholds shown in the Export tab."""
        if self.db is None or self.video_id is None:
            raise ValueError("Load a video before scoring a match.")
        video = dict(self.db.connection.execute("SELECT * FROM videos WHERE id=?", (self.video_id,)).fetchone())
        video.update(swarm_threshold=self.swarm_threshold.value(),
                     pollinator_1_threshold=self.pollinator_1_threshold.value(),
                     pollinator_2_threshold=self.pollinator_2_threshold.value())
        return match_scoreboard(video, [dict(robot) for robot in robots], [dict(event) for event in events])

    def _ruling_status_text(self, board: dict, flagged_count: int, robots) -> str:
        teams = {int(robot["id"]): robot["team_number"] for robot in robots}
        if board["match_type"] == "qualification":
            dq = board["ruling"]["qualification_disqualified_robot_ids"]
            if not dq:
                return "Qualification match: no team DQ recorded."
            return ("Qualification match: team " + ", ".join(str(teams[robot_id]) for robot_id in sorted(dq)) +
                    " DQ'd (0 match points, 0 Ranking Points); alliance partners keep their points (T601).")
        outcome = board["playoff_outcome"]
        if board["pending_referee_ruling"]:
            return ("PENDING REFEREE RULING: both alliances are DQ'd. Enter which alliance was disqualified first, "
                    "or a simultaneous DQ (T705). No winner is shown until then.")
        if outcome["resolved"] and outcome["tie"]:
            return "Playoff result: tie. The match is replayed (T705, 13.7.2)."
        if outcome["resolved"]:
            return f"Playoff result: {outcome['winner'].capitalize()} wins."
        if flagged_count:
            return f"Playoff result pending: {flagged_count} flagged event(s) need review, then mark the match final."
        return "Playoff result pending: mark the match final after reviewing all evidence."

    def build_rules_reference(self) -> str:
        resource_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        reference = resource_root / "docs" / "BIOBUZZ_RULES.md"
        if reference.is_file():
            return reference.read_text(encoding="utf-8")
        return "# BIOBUZZ rules unavailable\n\nSee the official [Competition Manual TU09](https://ftc-resources.firstinspires.org/ftc/archive/2027/game/cm-html/BIOBUZZ%20Competition%20Manual%20-%20TU09.htm)."

    def save_settings(self):
        folder = self.project_dir()
        if not folder:
            QMessageBox.information(self, "Open a project", "Project settings are saved with the active project."); return
        settings = {"model_path": self.model_path_edit.text().strip(), "reid_model_path": self.reid_path_edit.text().strip(), "ffmpeg_path": self.ffmpeg_path_edit.text().strip(),
                    "confidence": self.confidence_spin.value(), "sample_fps": int(self.sample_rate.currentText()),
                    "tracking_mode": self.tracking_mode.currentData()}
        (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        self.statusBar().showMessage("Settings saved in this project.")

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose local YOLO11n ONNX model", "", "ONNX model (*.onnx)")
        if path: self.model_path_edit.setText(path)

    def browse_reid(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose compatible local ReID model", "", "ReID model (*.onnx *.pt *.torchscript);;All files (*)")
        if path: self.reid_path_edit.setText(path)

    def browse_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose FFmpeg executable", "", "Executable (ffmpeg.exe);;All files (*)")
        if path: self.ffmpeg_path_edit.setText(path)

    def refresh_home(self):
        self.home_recent.clear()
        if not self.db or self.project_id is None:
            self._project_name = None
            self.home_project_label.setText("No project is open")
            self.home_project_path.setText("Create a project folder, or open an existing Control Vision project database.")
            self.home_empty.setText("Open a project to see its match videos.")
            self.home_empty.show()
            self._refresh_workflow()
            return
        project = self.db.connection.execute("SELECT name FROM projects WHERE id=?", (self.project_id,)).fetchone()
        self._project_name = project["name"]
        self.home_project_label.setText(project["name"])
        self.home_project_path.setText(str(self.db.path))
        for row in self.db.connection.execute("SELECT id,path,event_code,match_number,duration_s,analysis_status FROM videos WHERE project_id=? ORDER BY id DESC LIMIT 20", (self.project_id,)):
            state = {"ready": "Analysed", "unprocessed": "Not processed"}.get(row["analysis_status"], "Analysis did not finish")
            label = f"{Path(row['path']).name}  ·  {row['event_code'] or 'Practice'} {row['match_number'] or ''}  ·  {row['duration_s'] or 0:.0f}s  ·  {state}"
            item = QListWidgetItem(label); item.setData(Qt.ItemDataRole.UserRole, int(row['id'])); self.home_recent.addItem(item)
        self.home_empty.setText("No videos yet. Import a match video to start.")
        self.home_empty.setVisible(self.home_recent.count() == 0)
        self._refresh_workflow()

    def load_settings(self):
        folder = self.project_dir()
        if not folder: return
        path = folder / "settings.json"
        if not path.exists(): return
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
            self.model_path_edit.setText(settings.get("model_path", ""))
            mode_index = self.tracking_mode.findData(settings.get("tracking_mode", "yolo"))
            if mode_index >= 0: self.tracking_mode.setCurrentIndex(mode_index)
            self.reid_path_edit.setText(settings.get("reid_model_path", ""))
            self.ffmpeg_path_edit.setText(settings.get("ffmpeg_path", ""))
            self.confidence_spin.setValue(float(settings.get("confidence", 0.25)))
            self.sample_rate.setCurrentText(str(settings.get("sample_fps", 8)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Settings unavailable", f"Could not read this project's settings: {exc}")

    def start_zone_drawing(self):
        if self.video_id is None:
            QMessageBox.information(self, "Import a video", "Import or load a video before drawing zones."); return
        self._drawing_zone = True; self.field_map.draft = []; self.field_map.update()
        self.statusBar().showMessage(f"Draw a {self.zone_type.currentText()} zone: click at least three vertices on the map, then double-click to finish.")

    def start_roi_drawing(self):
        if self.video_id is None:
            QMessageBox.information(self, "Import a video", "Import or load a video before drawing a scoring-element ROI."); return
        self._mode = "roi"; self.canvas.roi_points = []
        if hasattr(self.canvas, "_last_frame"): self.canvas.show_frame(self.canvas._last_frame)
        self.statusBar().showMessage("Draw an image-space polygon around a HIVE CELL, FLOWER, or GARDEN; double-click to finish.")

    def start_named_roi_drawing(self, name: str):
        if self.video_id is None:
            QMessageBox.information(self, "Import a video", "Import or load a video before drawing scoreboard ROIs."); return
        self._mode = name; self.canvas.roi_points = []
        if hasattr(self.canvas, "_last_frame"): self.canvas.show_frame(self.canvas._last_frame)
        alliance = "red" if "red" in name else "blue"
        self.statusBar().showMessage(f"On the final results screen, outline only the {alliance} score digits. Click at least three corners, then double-click to save.")

    def finish_roi(self):
        if self._mode not in {"roi", "scoreboard_red_roi", "scoreboard_blue_roi"}: return
        name = "goal_roi" if self._mode == "roi" else self._mode
        if len(self.canvas.roi_points) < 3:
            self.statusBar().showMessage("The selected ROI needs at least three vertices."); return
        self.zones[name] = {"type": name, "points": self.canvas.roi_points.copy()}
        self.canvas.named_rois[name] = self.canvas.roi_points.copy()
        if name not in [self.zone_remove_combo.itemText(i) for i in range(self.zone_remove_combo.count())]:
            self.zone_remove_combo.addItem(name)
        self.db.save_calibration(self.video_id, self.corners, self.zones)
        self._mode = None
        if hasattr(self.canvas, "_last_frame"): self.canvas.show_frame(self.canvas._last_frame)
        message = "Scoring-element ROI saved. Ball-count changes remain flagged for human review." if name == "goal_roi" else f"{name.replace('_', ' ').title()} saved for local multi-frame OCR."
        self.statusBar().showMessage(message)
        self._refresh_workflow()

    def remove_selected_zone(self):
        name = self.zone_remove_combo.currentText()
        if not name or name not in self.zones: return
        del self.zones[name]
        if name.endswith("_roi"):
            self.canvas.named_rois.pop(name, None)
            if name == "goal_roi": self.canvas.roi_points = []
        self.field_map.zones = self.zones; self.field_map.update()
        self.db.save_calibration(self.video_id, self.corners, self.zones)
        self.zone_remove_combo.removeItem(self.zone_remove_combo.currentIndex())
        self._refresh_workflow()

    def zone_map_click(self, x: float, y: float):
        if not self._drawing_zone: return
        self.field_map.draft.append([round(x, 3), round(y, 3)]); self.field_map.update()

    def finish_zone(self):
        if not self._drawing_zone: return
        if len(self.field_map.draft) < 3:
            self.statusBar().showMessage("A zone needs at least three vertices."); return
        name, ok = QInputDialog.getText(self, "Name zone", f"Name this {self.zone_type.currentText()} zone")
        if not ok or not name.strip():
            self.field_map.draft = []; self.field_map.update(); self._drawing_zone = False; return
        clean_name = name.strip()
        if clean_name in self.zones:
            QMessageBox.warning(self, "Zone name already used", "Choose a unique zone name."); return
        if clean_name.endswith("_roi"):
            # Names ending in _roi are reserved for image-space ROIs.
            QMessageBox.warning(self, "Zone name reserved", "Field zone names cannot end in \"_roi\". Choose another name."); return
        self.zones[clean_name] = {"type": self.zone_type.currentText(), "points": self.field_map.draft.copy()}
        self.zone_remove_combo.addItem(clean_name)
        self.field_map.zones = self.zones; self.field_map.draft = []; self._drawing_zone = False
        self.db.save_calibration(self.video_id, self.corners, self.zones)
        self.field_map.update(); self.statusBar().showMessage(f"Saved zone '{clean_name}'.")
        self._refresh_workflow()

    def reuse_calibration(self):
        if self.video_id is None or self.metadata is None:
            QMessageBox.information(self, "Import a video", "Import a video before reusing calibration."); return
        rows = self.db.connection.execute("SELECT v.id,v.path,v.width,v.height,c.corners_json,c.zones_json FROM videos v JOIN calibrations c ON c.video_id=v.id WHERE v.project_id=? AND v.id<>? ORDER BY v.id DESC",
                                          (self.project_id, self.video_id)).fetchall()
        if not rows:
            QMessageBox.information(self, "No saved calibration", "This project has no other calibrated video yet."); return
        labels = [f"{Path(row['path']).name} ({row['width']}×{row['height']})" for row in rows]
        choice, ok = QInputDialog.getItem(self, "Reuse calibration", "Choose a camera setup", labels, 0, False)
        if not ok: return
        source = rows[labels.index(choice)]
        corners = json.loads(source["corners_json"])
        sx, sy = self.metadata["width"] / source["width"], self.metadata["height"] / source["height"]
        self.corners = [[round(x * sx, 2), round(y * sy, 2)] for x, y in corners]
        self.zones = json.loads(source["zones_json"] or "{}")
        for roi_name, roi in self.zones.items():
            if roi_name.endswith("_roi"):
                roi["points"] = [[round(x * sx, 2), round(y * sy, 2)] for x, y in roi.get("points", [])]
        self.zone_remove_combo.clear(); self.zone_remove_combo.addItems(self.zones.keys())
        self.db.save_calibration(self.video_id, self.corners, self.zones)
        self.field_map.zones = self.zones; self.canvas.named_rois = {key: val.get("points", []) for key, val in self.zones.items() if key.endswith("_roi")}; self.canvas.roi_points = self.zones.get("goal_roi", {}).get("points", []); self.field_map.update()
        if hasattr(self.canvas, "_last_frame"): self.canvas.show_frame(self.canvas._last_frame)
        self.statusBar().showMessage("Corners and user-drawn zones copied. Confirm they align with this camera view before processing.")
        self._refresh_workflow()

    def new_project(self):
        if _video_switch_blocked(self): return
        folder = QFileDialog.getExistingDirectory(self, "Choose project folder")
        if not folder:
            return
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New project", "Project name")
        if not ok or not name.strip():
            return
        try:
            if self.db: self.db.close()
            self.db = ProjectDatabase(Path(folder) / "control_vision.sqlite")
            self.project_id = self.db.add_project(name.strip())
            self.reset_video_state()
            (Path(folder) / "run_logs").mkdir(exist_ok=True)
            self.load_settings()
            self.statusBar().showMessage(f"Project ready: {name.strip()}")
            self.refresh_home()
        except Exception as exc:
            # The previous database is already closed; never keep using it.
            if self.db: self.db.close()
            self.db, self.project_id = None, None
            self._clear_closed_project(f"Could not create project: {exc}")
            QMessageBox.critical(self, "Could not create project", str(exc))

    def open_project(self):
        if _video_switch_blocked(self): return
        selected, _ = QFileDialog.getOpenFileName(self, "Open Control Vision project", "", "Control Vision project (*.sqlite *.db)")
        if not selected:
            return
        try:
            if self.db: self.db.close()
            self.db = ProjectDatabase(Path(selected))
            project = self.db.connection.execute("SELECT id,name FROM projects ORDER BY id LIMIT 1").fetchone()
            if project is None:
                raise ValueError("This database does not contain a Control Vision project.")
            self.project_id = project["id"]
            self.reset_video_state()
            self.load_settings()
            (self.project_dir() / "run_logs").mkdir(exist_ok=True)
            self.statusBar().showMessage(f"Project opened: {project['name']}")
            self.refresh_home()
        except Exception as exc:
            self.db = None
            self.project_id = None
            self._clear_closed_project(f"Could not open project: {exc}")
            QMessageBox.critical(self, "Could not open project", str(exc))

    def _clear_closed_project(self, message: str):
        """After a failed project switch the previous project is closed: show no project or video."""
        self.reset_video_state()
        self._show_status(message, "danger")

    def reset_video_state(self):
        """Clear match-specific UI state when switching projects."""
        if self.capture:
            self.capture.release()
        self.capture = None
        self.video_id = self.video_path = self.metadata = None
        self.scoreboard_ocr = None
        self.corners = []
        self.zones = {}
        self.seed_points = []
        self.robot_labels = []
        self.tracks = {}
        self.cycles = []
        self.events = []
        self.photo_rows = []
        self.current_time = self.duration = 0.0
        self.canvas.roi_points = []
        self.canvas.named_rois = {}
        self.canvas.clear()
        self.field_map.zones = {}
        self.field_map.tracks = {}
        self.field_map.current_time = 0.0
        self.field_map.update()
        self.zone_remove_combo.clear()
        self.official_red.clear()
        self.official_blue.clear()
        self.show_scoreboard_ocr(None)
        self.load_match_ruling()
        self.swarm_threshold.setValue(16); self.pollinator_1_threshold.setValue(4); self.pollinator_2_threshold.setValue(7)
        self.cycles_table.setRowCount(0)
        self.stats_table.setRowCount(0)
        self.review_table.setRowCount(0)
        self.scoring_table.setRowCount(0)
        self.clip_list.clear()
        self._set_results_enabled(False)
        self._show_status("Project opened. Load or import a match video to begin.")
        self.slider.setValue(0)
        self.refresh_result_tables()
        self.refresh_home()

    def _set_results_enabled(self, enabled: bool):
        """Hide saved analysis while a rerun is active or has failed."""
        if not enabled:
            self.tracks = {}
            self.cycles = []
            self.events = []
            self.photo_rows = []
            self.field_map.tracks = {}
            self.field_map.update()
            self.cycles_table.setRowCount(0)
            self.stats_table.setRowCount(0)
            self.review_table.setRowCount(0)
            self.scoring_table.setRowCount(0)
            self.clip_list.clear()
            if 2 <= self.tabs.currentIndex() <= 6:
                self.tabs.setCurrentIndex(1)
            self._set_review_badge(0)
        for index in range(2, 7):
            self.tabs.setTabEnabled(index, enabled)
        self._sync_nav()

    def _results_current(self) -> bool:
        if not self.db or self.video_id is None:
            return False
        row = self.db.connection.execute(
            "SELECT analysis_status FROM videos WHERE id=?", (self.video_id,)).fetchone()
        if row and row["analysis_status"] == "ready":
            return True
        message = ("Process this video before reviewing or exporting results."
                   if row and row["analysis_status"] == "unprocessed"
                   else "The last analysis did not finish. Process this video successfully before reviewing or exporting results.")
        QMessageBox.warning(self, "Analysis unavailable", message)
        return False

    def import_video(self):
        if _video_switch_blocked(self): return
        if not self.db or self.project_id is None:
            QMessageBox.information(self, "Create a project", "Create or open a project before importing a video.")
            return
        selected, _ = QFileDialog.getOpenFileName(self, "Import match video", "", "Videos (*.mp4 *.mov *.mkv)")
        if not selected:
            return
        path = Path(selected)
        try:
            metadata = probe_video(path)
            video_warnings = list(metadata.get("warnings", []))
            if video_warnings:
                QMessageBox.warning(self, "Video quality warning", "\n".join(video_warnings) + "\n\nYou can still import this video.")
            event_code, ok = QInputDialog.getText(self, "Match details", "Event code (optional)")
            if not ok: return
            match_number, ok = QInputDialog.getText(self, "Match details", "Match number (optional, e.g. Q12)")
            if not ok: return
            cap = cv2.VideoCapture(str(path)); ok, frame = cap.read(); cap.release()
            if not ok:
                raise VideoError(f"Could not read the first frame from {path.name}.")
            video_id = self.db.add_video(self.project_id, path, metadata, event_code.strip() or None, match_number.strip() or None)
        except Exception as exc:
            QMessageBox.critical(self, "Video import failed", str(exc)); return
        if self.capture: self.capture.release()
        self.video_id, self.video_path, self.metadata = video_id, path, metadata
        self.scoreboard_ocr = None
        self.corners.clear(); self.seed_points.clear(); self.robot_labels.clear(); self.tracks.clear(); self.zones.clear(); self.cycles.clear(); self.events = []
        self.zone_remove_combo.clear(); self.official_red.clear(); self.official_blue.clear()
        self.show_scoreboard_ocr(None)
        self.load_match_ruling()
        self.swarm_threshold.setValue(int(self.scoring_config["ranking_points"]["thresholds_by_event_type"]["standard"]["swarm_leave_park_points"]))
        self.pollinator_1_threshold.setValue(int(self.scoring_config["ranking_points"]["thresholds_by_event_type"]["standard"]["pollinator_1_hive_tips"]))
        self.pollinator_2_threshold.setValue(int(self.scoring_config["ranking_points"]["thresholds_by_event_type"]["standard"]["pollinator_2_hive_tips"]))
        self.canvas.roi_points = []
        self.canvas.named_rois = {}
        self.field_map.zones = {}; self.field_map.tracks = {}; self.field_map.update()
        self._set_results_enabled(False)
        self.duration = float(metadata["duration_s"])
        self.capture = cv2.VideoCapture(str(path))
        self.canvas.show_frame(frame)
        self.slider.setValue(0)
        message = f"{path.name}: {metadata['width']}×{metadata['height']}, {metadata['fps']:.2f} fps, {self.duration:.1f} s"
        if video_warnings: message += " — warning: " + " ".join(video_warnings)
        self._show_status(message, "pending" if video_warnings else "info")
        self.statusBar().showMessage("Video imported. Select the field corners in this order: top-left, top-right, bottom-right, bottom-left.")
        self.refresh_home()

    def load_saved_video(self):
        if _video_switch_blocked(self): return
        if not self.db or self.project_id is None:
            QMessageBox.information(self, "Open a project", "Open a project before loading one of its saved videos.")
            return
        videos = self.db.connection.execute(
            "SELECT id,path,duration_s,width,height,official_red,official_blue,red_disqualified,blue_disqualified,swarm_threshold,pollinator_1_threshold,pollinator_2_threshold,scoreboard_ocr_json,tracking_mode FROM videos WHERE project_id=? ORDER BY id DESC", (self.project_id,)
        ).fetchall()
        if not videos:
            QMessageBox.information(self, "No videos", "This project has no imported videos yet.")
            return
        choices = [f"#{row['id']} {Path(row['path']).name} — {row['duration_s'] or 0:.1f}s" for row in videos]
        from PySide6.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(self, "Load saved video", "Video", choices, 0, False)
        if not ok: return
        video = videos[choices.index(choice)]
        self.load_saved_video_id(int(video["id"]))

    def load_saved_video_id(self, video_id: int | None):
        if _video_switch_blocked(self): return
        if not self.db or self.project_id is None or video_id is None: return
        video = self.db.connection.execute(
            "SELECT id,path,duration_s,width,height,official_red,official_blue,red_disqualified,blue_disqualified,swarm_threshold,pollinator_1_threshold,pollinator_2_threshold,scoreboard_ocr_json,tracking_mode,analysis_status FROM videos WHERE project_id=? AND id=?",
            (self.project_id, int(video_id)),
        ).fetchone()
        if video is None: return
        mode_index = self.tracking_mode.findData(video["tracking_mode"] or "yolo")
        if mode_index >= 0: self.tracking_mode.setCurrentIndex(mode_index)
        path = Path(video["path"])
        if not path.is_file():
            QMessageBox.warning(self, "Video unavailable", f"The saved video path no longer exists: {path}")
            return
        try:
            metadata = probe_video(path)
            cap = cv2.VideoCapture(str(path)); read_ok, frame = cap.read()
            if not read_ok:
                cap.release(); raise VideoError(f"Could not read the first frame from {path.name}.")
        except Exception as exc:
            QMessageBox.critical(self, "Could not load video", str(exc)); return
        if self.capture: self.capture.release()
        self.capture = cap
        self.video_id, self.video_path, self.metadata = video["id"], path, metadata
        results_available = video["analysis_status"] == "ready"
        self._set_results_enabled(results_available)
        self.duration = float(metadata["duration_s"]); self.current_time = 0
        self.official_red.setText(str(video["official_red"]) if video["official_red"] is not None else "")
        self.official_blue.setText(str(video["official_blue"]) if video["official_blue"] is not None else "")
        self.load_match_ruling()
        self.swarm_threshold.setValue(int(video["swarm_threshold"])); self.pollinator_1_threshold.setValue(int(video["pollinator_1_threshold"])); self.pollinator_2_threshold.setValue(int(video["pollinator_2_threshold"]))
        calibration = self.db.connection.execute("SELECT corners_json FROM calibrations WHERE video_id=?", (self.video_id,)).fetchone()
        self.corners = json.loads(calibration["corners_json"]) if calibration else []
        zones_row = self.db.connection.execute("SELECT zones_json FROM calibrations WHERE video_id=?", (self.video_id,)).fetchone()
        self.zones = json.loads(zones_row["zones_json"] or "{}") if zones_row else {}
        self.field_map.zones = self.zones
        self.canvas.roi_points = self.zones.get("goal_roi", {}).get("points", [])
        self.canvas.named_rois = {key: val.get("points", []) for key, val in self.zones.items() if key.endswith("_roi")}
        self.scoreboard_ocr = json.loads(video["scoreboard_ocr_json"]) if video["scoreboard_ocr_json"] else None
        self.show_scoreboard_ocr(self.scoreboard_ocr)
        self.zone_remove_combo.clear(); self.zone_remove_combo.addItems(self.zones.keys())
        self.events = self.db.events_for_video(self.video_id) if results_available else []
        self.photo_rows = self.db.photos_for_video(self.video_id) if results_available else []
        robots = self.db.connection.execute("SELECT id,team_number,alliance,seed_x,seed_y FROM robots WHERE video_id=? ORDER BY id", (self.video_id,)).fetchall()
        self.seed_points = [(row["id"], row["seed_x"], row["seed_y"]) for row in robots if row["seed_x"] is not None and row["seed_y"] is not None]
        self.robot_labels = [f"{row['team_number']} ({row['alliance']})" for row in robots]
        self.canvas.show_frame(frame, [(int(x), int(y), str(index + 1)) for index, (x, y) in enumerate(self.corners)])
        self.refresh_home()
        warning_note = (" Warning: " + " ".join(metadata["warnings"])) if metadata.get("warnings") else ""
        self._show_status(f"Loaded {path.name}. Corners: {len(self.corners)}/4; seeded identities: {len(self.seed_points)}.{warning_note}", "pending" if warning_note else "info")
        rows = self.db.connection.execute("SELECT robot_id,t_s,x_ft,y_ft FROM tracks WHERE robot_id IN (SELECT id FROM robots WHERE video_id=?) ORDER BY frame", (self.video_id,)).fetchall() if results_available else []
        track_by_id = {}
        for row in rows: track_by_id.setdefault(row["robot_id"], []).append((row["t_s"], row["x_ft"], row["y_ft"]))
        self.tracks = {i: values for i, values in enumerate(track_by_id.values())}
        self.field_map.tracks = self.tracks; self.field_map.current_time = 0; self.field_map.update()
        self.refresh_result_tables()
        if not results_available:
            unprocessed = video["analysis_status"] == "unprocessed"
            # A run that did not finish is an error, like the Process step's failed state.
            tone = "danger" if not unprocessed else ("pending" if warning_note else "info")
            self._show_status((
                "This video has not been processed. Set up the field and run analysis to create results."
                if unprocessed
                else "The last analysis did not finish. Saved results are hidden until this video is processed successfully."
            ) + warning_note, tone)

    def select_corner(self):
        if self.video_path is None: return
        self._mode = "corner"
        self.statusBar().showMessage(f"Click corner {len(self.corners) % 4 + 1} of 4 in order (top-left, top-right, bottom-right, bottom-left).")

    def seed_robot(self):
        if self.video_path is None: return
        if len(self.seed_points) >= 4:
            QMessageBox.information(self, "Robot limit", "Up to four robots can be seeded for a match."); return
        self._mode = "robot"
        self.statusBar().showMessage("Click a robot near its ground-contact point to assign its team and alliance.")

    def canvas_click(self, x, y):
        if self._mode == "corner":
            if len(self.corners) >= 4:
                # Selecting corners again starts a new calibration.
                self.corners.clear()
            self.corners.append([x, y]); self._mode = None
            annotations = [(int(cx), int(cy), f"{i+1}") for i, (cx, cy) in enumerate(self.corners)]
            if self.video_path:
                cap = cv2.VideoCapture(str(self.video_path)); ok, frame = cap.read(); cap.release()
                if ok: self.canvas.show_frame(frame, annotations)
            if len(self.corners) == 4:
                try:
                    from control_vision.pipeline.homography import matrix_from_corners, reprojection_error_px
                    matrix_from_corners(self.corners)
                    self.db.save_calibration(self.video_id, self.corners, self.zones)
                    self.field_map.update()
                    self._show_status(f"Calibration saved. Four-corner interpolation residual: {reprojection_error_px(self.corners):.2f} px (this is not an independent accuracy check).")
                except Exception as exc:
                    self.corners.clear()
                    self._show_status(f"Calibration not saved: {exc} Use Calibrate to select the 4 corners again.", "danger")
                    if hasattr(self.canvas, "_last_frame"): self.canvas.show_frame(self.canvas._last_frame)
                    QMessageBox.warning(self, "Invalid calibration", str(exc))
        elif self._mode == "robot":
            dialog = IdentityDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                robot_id = self.db.add_robot(self.video_id, dialog.team.value(), dialog.alliance.currentText(), x, y)
                self.seed_points.append((robot_id, x, y))
                self.robot_labels.append(f"{dialog.team.value()} ({dialog.alliance.currentText()})")
                self._show_status("Robot identities: " + ", ".join(self.robot_labels))
                if self.capture:
                    self.capture.set(cv2.CAP_PROP_POS_MSEC, 0); ok, frame = self.capture.read()
                    if ok: self.canvas.show_frame(frame, [(sx, sy, label) for (_, sx, sy), label in zip(self.seed_points, self.robot_labels)])
            self._mode = None
        elif self._mode == "roi" or self._mode in {"scoreboard_red_roi", "scoreboard_blue_roi"}:
            self.canvas.roi_points.append([x, y])
            if hasattr(self.canvas, "_last_frame"): self.canvas.show_frame(self.canvas._last_frame)
        self._refresh_workflow()

    def process(self):
        if not self.db or not self.video_path or len(self.corners) != 4 or not self.seed_points:
            QMessageBox.information(self, "Setup incomplete", "Create a project, import a video, select four corners, and seed at least one robot first.")
            return
        mode = self.tracking_mode.currentData()
        model_path = self.model_path_edit.text().strip()
        if mode == "yolo":
            if not model_path:
                model_path, _ = QFileDialog.getOpenFileName(self, "Select local YOLO11n ONNX model", "models", "ONNX model (*.onnx)")
                if not model_path: return
                self.model_path_edit.setText(model_path)
            if not Path(model_path).is_file():
                QMessageBox.warning(self, "Model not found", f"No model file exists at {model_path}. Choose Seeded optical flow for model-free tracking or select a trained local YOLO model in Settings.")
                return
        settings_path = self.project_dir() / "settings.json"
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            settings = {}
        log_dir = Path(self.db.path).parent / "run_logs"
        log_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logger = logging.getLogger(f"control_vision.{stamp}"); self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir / f"run-{stamp}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); self.logger.addHandler(handler)
        self.logger.info("Video probed: %s; %s", self.video_path, self.metadata)
        self.logger.info("Zones: %s; sample fps: %s; confidence threshold: %s; tracking mode: %s", self.zones, self.sample_rate.currentText(), self.confidence_spin.value(), mode)
        self.db.set_analysis_status(self.video_id, "running")
        self._set_results_enabled(False)
        self.process_btn.setEnabled(False)
        photos_dir = self.project_dir() / "photos"
        alliances = {row["id"]: row["alliance"] for row in self.db.robots_for_video(self.video_id)}
        self._worker = ProcessingWorker((self.video_path, Path(model_path), self.corners.copy(), self.seed_points.copy(),
                                         float(self.sample_rate.currentText()), self.logger, photos_dir, self.confidence_spin.value(),
                                         self.zones.get("goal_roi", {}).get("points"), self.reid_path_edit.text().strip(), mode, alliances))
        self._thread = QThread(self); self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.show_progress)
        self._worker.done.connect(self.processing_done)
        self._worker.failed.connect(self.processing_failed)
        self._worker.done.connect(self._thread.quit); self._worker.failed.connect(self._thread.quit)
        self._processing_active = True
        self._show_status("Starting analysis…")
        self.status_progress.setValue(0); self.status_progress.show()
        self._refresh_workflow()
        try:
            self._thread.start()
        except Exception as exc:
            self.processing_failed(f"Could not start analysis: {exc}")

    def show_progress(self, stage, percent, detail):
        self._show_status(f"{stage} — {percent:.0f}% — {detail}")
        self.status_progress.setValue(max(0, min(100, round(percent))))
        self._step_widgets["process"][1].setText(f"{stage} — {percent:.0f}%")

    def processing_done(self, rows):
        (track_rows, _, _), tracking_mode = rows
        if not track_rows:
            # The seeded preview suspends uncertain tracks and may record
            # nothing; an empty run must not be presented as an analysis.
            method = "The seeded preview" if tracking_mode == "seeded_optical_flow" else "Tracking"
            self.processing_failed(
                f"{method} did not record a confident position for any seeded robot, so no results were saved "
                "and this video is not analysed. Seed each robot on a clear frame where it is fully visible, "
                "or use a trained YOLO model.")
            return
        try:
            self._store_processing_results(rows)
        except Exception as exc:
            logger = getattr(self, "logger", None)
            if logger: logger.exception("Saving analysis results failed")
            self.processing_failed(f"Could not save the analysis results: {exc}")
            return
        self.process_btn.setEnabled(True)
        if self.capture: self.timer.start()
        self._processing_active = False
        self.status_progress.hide()
        self._refresh_workflow()

    def _store_processing_results(self, rows):
        (track_rows, photos, roi_events), tracking_mode = rows
        self.db.set_tracking_mode(self.video_id, tracking_mode)
        robot_ids = [seed[0] for seed in self.seed_points]
        self.db.replace_tracks(track_rows, robot_ids)
        self.db.replace_photos(robot_ids, photos)
        self.photo_rows = photos
        saved_mode = self.db.connection.execute("SELECT tracking_mode FROM videos WHERE id=?", (self.video_id,)).fetchone()
        preview_only = bool(saved_mode and saved_mode["tracking_mode"] == "seeded_optical_flow")
        cycles, cycle_events = ([], []) if preview_only else count_zone_cycles(track_rows, self.zones)
        self.db.replace_analysis(robot_ids, cycles, cycle_events + roi_events)
        self.db.set_analysis_status(self.video_id, "ready")
        self._set_results_enabled(True)
        self.cycles = self.db.cycles_for_video(self.video_id)
        self.events = self.db.events_for_video(self.video_id)
        by_id = {}
        for robot_id, frame, t_s, x, y, conf in track_rows:
            by_id.setdefault(robot_id, []).append((t_s, x, y))
        self.tracks = {i: points for i, points in enumerate(by_id.values())}
        self.field_map.tracks = self.tracks; self.field_map.update()
        tracking_method = "seeded optical flow" if tracking_mode == "seeded_optical_flow" else "YOLO tracking"
        self.log_view.addItem(f"{tracking_method} complete: {len(track_rows):,} mapped robot samples; {len(self.cycles)} zone-trip cycles; {len(photos)} robot photos")
        if tracking_mode == "seeded_optical_flow":
            self.log_view.addItem("Accuracy note: optical-flow is a trail preview only. Cycle counts and movement statistics are omitted; verify every identity and position before any use as evidence.")
        message = f"Analysis complete. Samples: {len(track_rows):,}; cycle/event candidates: {len(self.events)}. Review points and statuses before reporting scores."
        missing = missing_robot_labels(self.db.robots_for_video(self.video_id), robot_ids, track_rows)
        if missing:
            note = ("No confident position was recorded for " + ", ".join(missing) +
                    "; their trails and statistics are empty.")
            self.log_view.addItem(note)
            message += " " + note
        self._show_status(message, "pending" if missing else "info")
        self.refresh_result_tables()

    def refresh_result_tables(self):
        if not self.db or self.video_id is None: return
        status = self.db.connection.execute(
            "SELECT analysis_status FROM videos WHERE id=?", (self.video_id,)).fetchone()
        if status and status["analysis_status"] != "ready":
            self._set_results_enabled(False)
            return
        self.cycles = self.db.cycles_for_video(self.video_id)
        self.events = self.db.events_for_video(self.video_id)
        robots = self.db.robots_for_video(self.video_id)
        self.review_robot.clear()
        self.review_robot.addItem("Unassigned — choose a robot only if the video supports it", None)
        for robot in robots:
            self.review_robot.addItem(f"{robot['team_number']} ({robot['alliance']})", robot["id"])
        self.cycles_table.setRowCount(len(self.cycles))
        for row_index, cycle in enumerate(self.cycles):
            values = [cycle["team_number"], cycle["alliance"], f"{cycle['start_t_s']:.2f}", f"{cycle['end_t_s']:.2f}", cycle["pickup_zone"], cycle["scoring_zone"]]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value)); self.cycles_table.setItem(row_index, col, item)
            self.cycles_table.item(row_index, 0).setData(Qt.ItemDataRole.UserRole, cycle["start_t_s"])
        track_rows = [tuple(row) for row in self.db.tracks_for_video(self.video_id)]
        saved_mode = self.db.connection.execute(
            "SELECT tracking_mode FROM videos WHERE id=?", (self.video_id,)).fetchone()
        preview_only = bool(saved_mode and saved_mode["tracking_mode"] == "seeded_optical_flow")
        stats = {} if preview_only else robot_statistics(track_rows, [dict(row) for row in self.cycles], self.zones)
        self.load_match_ruling()
        self.stats_table.setRowCount(len(robots))
        for row_index, robot in enumerate(robots):
            values = stats.get(robot["id"], {"cycles": "Preview only" if preview_only else 0,
                                                "cycle_time_avg_s": None if preview_only else 0,
                                                "top_speed_ft_s": None if preview_only else 0,
                                                "distance_ft": None if preview_only else 0})
            zone_seconds = values.get("zone_time_s", {})
            zone_summary = "Unavailable in preview mode" if preview_only else ", ".join(f"{name}: {seconds:.1f}" for name, seconds in zone_seconds.items())
            display = [robot["team_number"], robot["alliance"], values["cycles"], "—" if preview_only else f"{values['cycle_time_avg_s']:.2f}", "—" if preview_only else f"{values['top_speed_ft_s']:.2f}", "—" if preview_only else f"{values['distance_ft']:.2f}", zone_summary, "", ""]  # points columns: update_score_comparison
            for col, value in enumerate(display): self.stats_table.setItem(row_index, col, QTableWidgetItem(str(value)))
        flagged = [event for event in self.events if event["status"] == "flagged"]
        self.review_table.setRowCount(len(flagged))
        # The review controls describe one event; require a fresh selection
        # after every rebuild so stale values are never applied to another row.
        self.review_table.setCurrentCell(-1, -1)
        clips_by_event = {row["scoring_event_id"]: row for row in self.db.clips_for_video(self.video_id)}
        for row_index, event in enumerate(flagged):
            clip = clips_by_event.get(event["id"])
            event_label = self.event_types.get(event["event_type"], {}).get("label", "Unclassified / other")
            phase, remaining = classify_match_time(float(event["t_s"]), self.scoring_config["match_periods_s"])
            values = [f"{event['t_s']:.2f}s {phase} ({remaining:.0f}s left)", event["team_number"], event["alliance"], event["event_alliance"], event_label, event["points"], theme.confidence_label(event["confidence"]), theme.status_label(event["status"]), Path(clip["path"]).name if clip else "Not generated"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value)); self.review_table.setItem(row_index, col, item)
            self.review_table.item(row_index, 0).setData(Qt.ItemDataRole.UserRole, event["id"])
            self._paint_event_row(self.review_table, row_index, event["status"], event["confidence"])
        self._set_review_badge(len(flagged))
        self.scoring_table.setRowCount(len(self.events))
        for row_index, event in enumerate(self.events):
            event_label = self.event_types.get(event["event_type"], {}).get("label", "Unclassified / other")
            phase, remaining = classify_match_time(float(event["t_s"]), self.scoring_config["match_periods_s"])
            values = [f"{event['t_s']:.2f}s {phase} ({remaining:.0f}s left)", event["team_number"], event["alliance"], event["event_alliance"], event_label, event["points"], theme.confidence_label(event["confidence"]), theme.status_label(event["status"])]
            for col, value in enumerate(values): self.scoring_table.setItem(row_index, col, QTableWidgetItem(str(value)))
            self.scoring_table.item(row_index, 0).setData(Qt.ItemDataRole.UserRole, event["t_s"])
            self._paint_event_row(self.scoring_table, row_index, event["status"], event["confidence"])
        self.refresh_clips()
        self.update_score_comparison()

    def jump_to_cycle(self, row: int, column: int):
        item = self.cycles_table.item(row, 0)
        if item: self.seek(round(float(item.data(Qt.ItemDataRole.UserRole)) / max(self.duration, 0.001) * 1000))

    def jump_to_event(self, row: int, column: int):
        item = self.scoring_table.item(row, 0)
        if item: self.seek(round(float(item.data(Qt.ItemDataRole.UserRole)) / max(self.duration, 0.001) * 1000))

    def add_manual_event(self):
        if not self.db or self.video_id is None or self.review_robot.count() <= 1:
            QMessageBox.information(self, "Set up robot identities", "Seed at least one robot before adding a scoring event."); return
        if self.review_robot.currentData() is None:
            QMessageBox.information(self, "Choose a robot", "Choose the attributed robot in Review before adding an event at the current video time."); return
        self.db.add_scoring_event(int(self.review_robot.currentData()), self.current_time)
        self.refresh_result_tables()
        self.statusBar().showMessage(f"Added a flagged scoring candidate at {self.current_time:.2f}s. Classify it after reviewing the video.")

    def review_selected(self, status: str):
        row = self.review_table.currentRow()
        if row < 0 or not self.db: return
        event_id = self.review_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        robot_id = self.review_robot.currentData()
        try:
            self.db.review_event(event_id, status, self.review_points.value(), robot_id, self.review_type.currentData(), self.review_confidence.currentText(), self.review_alliance.currentData())
        except ValueError as exc:
            QMessageBox.warning(self, "Review not saved", str(exc)); return
        self.refresh_result_tables()

    def review_type_changed(self, index: int):
        event_type = self.review_type.currentData()
        if event_type in self.event_types:
            self.review_points.setValue(int(self.event_types[event_type]["points"]))
            robot_id = self.review_robot.currentData()
            robot = self.db.connection.execute("SELECT alliance FROM robots WHERE id=?", (robot_id,)).fetchone() if self.db and robot_id is not None else None
            if robot and self.event_types[event_type].get("recipient") == "opponent":
                recipient = "blue" if robot["alliance"] == "red" else "red"
                self.review_alliance.setCurrentIndex(self.review_alliance.findData(recipient))
            selected = self.review_table.currentRow()
            event = None
            if 0 <= selected < self.review_table.rowCount():
                event_id = self.review_table.item(selected, 0).data(Qt.ItemDataRole.UserRole)
                event = next((entry for entry in self.events if entry["id"] == event_id), None)
            if event and event_type.startswith("flower_") and float(event["t_s"]) < float(self.scoring_config["match_periods_s"]["flower_unlock_elapsed_s"]):
                self.review_rule_hint.setText("TU09 G410: FLOWER scoring starts with 60 seconds left in TELEOP. This early achievement can still score, but each early NECTAR may receive a 20-point Major Foul; verify the official call and add the penalty separately if recorded.")
            elif event_type in {"minor_foul", "major_foul"}:
                self.review_rule_hint.setText("Foul credits are awarded to the opponent alliance. Preserve the offending robot attribution and set the recipient alliance separately. Only record referee-assessed fouls.")
            else:
                self.review_rule_hint.setText("Scoring recipient can differ from the attributed robot. Confirm achievement evidence and receiver from the rule criteria before marking confirmed.")

    def load_review_selection(self, row: int, column: int, previous_row: int, previous_column: int):
        if row < 0 or row >= self.review_table.rowCount(): return
        item = self.review_table.item(row, 0)
        if not item: return
        event_id = item.data(Qt.ItemDataRole.UserRole)
        event = next((entry for entry in self.events if entry["id"] == event_id), None)
        if not event: return
        self.review_type.blockSignals(True)
        index = self.review_type.findData(event["event_type"])
        self.review_type.setCurrentIndex(index if index >= 0 else 0)
        self.review_type.blockSignals(False)
        robot_index = 0 if event["robot_id"] is None else self.review_robot.findData(event["robot_id"])
        self.review_robot.setCurrentIndex(robot_index if robot_index >= 0 else 0)
        self.review_points.setValue(int(event["points"]))
        self.review_confidence.setCurrentText(event["confidence"])
        alliance_index = self.review_alliance.findData(event["scoring_alliance"])
        self.review_alliance.setCurrentIndex(alliance_index if alliance_index >= 0 else 0)

    def update_score_comparison(self, *_):
        if not self.db or self.video_id is None:
            self.score_compare.setText("Video totals include confirmed events only.")
            theme.set_tone(self.score_compare, "info")
            self.ruling_status.setText(""); theme.set_tone(self.ruling_status, None); return
        events = self.db.events_for_video(self.video_id)
        robots = [dict(row) for row in self.db.robots_for_video(self.video_id)]
        try:
            board = self.current_scoreboard(events, robots)
        except ValueError as exc:
            self.score_compare.setText(f"Score unavailable: {exc}")
            theme.set_tone(self.score_compare, "danger")
            self.ruling_status.setText(str(exc)); theme.set_tone(self.ruling_status, "danger"); return
        flagged_count = sum(event["status"] == "flagged" for event in events)
        self.ruling_status.setText(self._ruling_status_text(board, flagged_count, robots))
        theme.set_tone(self.ruling_status, "pending" if board["pending_referee_ruling"] else None)
        has_tracks = self.db.connection.execute("SELECT 1 FROM tracks WHERE robot_id IN (SELECT id FROM robots WHERE video_id=?) LIMIT 1", (self.video_id,)).fetchone()
        if not events and not has_tracks:
            self.score_compare.setText("No analysis has run for this video yet.")
            theme.set_tone(self.score_compare, "info"); return
        theme.set_tone(self.score_compare, "pending" if flagged_count else "info")
        totals = board["alliance_points"]
        playoff = board["match_type"] == "playoff"
        dq_ids = board["ruling"]["qualification_disqualified_robot_ids"]
        dq_alliances = board["ruling"]["playoff_disqualified"]
        if self.stats_table.rowCount() == len(robots):
            for row_index, robot in enumerate(robots):
                robot_id = int(robot["id"])
                points = str(board["team_points"].get(robot_id, 0))
                if robot_id in dq_ids or robot["alliance"] in dq_alliances:
                    points += " (DQ)"
                ranking = "— (playoff)" if playoff else str(board["team_ranking_points"].get(robot_id, 0))
                self.stats_table.setItem(row_index, 7, QTableWidgetItem(points))
                self.stats_table.setItem(row_index, 8, QTableWidgetItem(ranking))
        if playoff:
            state = ("UNREVIEWED — final score pending" if flagged_count else
                     "Marked final" if board["match_resolved"] else "Not marked final")
        else:
            state = ("UNREVIEWED — final score pending" if flagged_count else
                     "Reviewed" if board["match_resolved"] else "No scoring events reviewed")
        parts = [f"{state}. Video total — red {totals['red']}, blue {totals['blue']}, match {sum(totals.values())}; flagged events: {flagged_count}."]
        for alliance, edit in (("red", self.official_red), ("blue", self.official_blue)):
            raw = edit.text().strip()
            if raw.isdigit(): parts.append(f"{alliance.capitalize()} official {int(raw)}; difference {totals[alliance] - int(raw):+d}")
        self.score_compare.setText("  ".join(parts))
        if playoff:
            self.ranking_summary.setText("Playoff match: teams do not earn Ranking Points (13.7); "
                                         "the winner advances in the bracket.")
            return
        rp = board["ranking_points"]
        self.ranking_summary.setText(
            f"Ranking Points per team (standard thresholds: SWARM ≥{board['swarm_threshold']} Leave+Park points; "
            f"POLLINATOR 1 ≥{board['pollinator_thresholds'][0]} HIVE TIPS; POLLINATOR 2 ≥{board['pollinator_thresholds'][1]}): "
            f"Red {rp['red']['total']} [SWARM {rp['red']['SWARM']}, P1 {rp['red']['POLLINATOR_1']}, P2 {rp['red']['POLLINATOR_2']}, Win/Tie {rp['red']['WIN_TIE']}]; "
            f"Blue {rp['blue']['total']} [SWARM {rp['blue']['SWARM']}, P1 {rp['blue']['POLLINATOR_1']}, P2 {rp['blue']['POLLINATOR_2']}, Win/Tie {rp['blue']['WIN_TIE']}].")

    def save_official_scores(self):
        if not self.db or self.video_id is None: return
        red, blue = self.official_red.text().strip(), self.official_blue.text().strip()
        if (red and not red.isdigit()) or (blue and not blue.isdigit()):
            QMessageBox.warning(self, "Invalid official score", "Enter whole-number red and blue totals, or leave a field blank."); return
        # The DQ ruling is saved separately by save_match_ruling; None keeps it.
        self.db.set_official_scores(self.video_id, int(red) if red else None, int(blue) if blue else None,
                                    None, self.swarm_threshold.value(), self.pollinator_1_threshold.value(), self.pollinator_2_threshold.value())
        self.update_score_comparison()

    def start_scoreboard_ocr(self):
        if not self.video_path or not self.video_id:
            QMessageBox.information(self, "Load a video", "Import or load a video before reading its scoreboard."); return
        red = self.zones.get("scoreboard_red_roi", {}).get("points")
        blue = self.zones.get("scoreboard_blue_roi", {}).get("points")
        if not red or not blue:
            QMessageBox.information(self, "Set scoreboard ROIs", "Draw the red and blue scoreboard ROIs on the video frame first."); return
        self.read_scoreboard_btn.setEnabled(False)
        self.use_ocr_btn.setEnabled(False)
        self._ocr_worker = ScoreboardOcrWorker(self.video_path, red, blue)
        self._ocr_thread = QThread(self)
        self._ocr_worker.moveToThread(self._ocr_thread)
        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.progress.connect(self.show_ocr_progress)
        self._ocr_worker.done.connect(self.scoreboard_ocr_done)
        self._ocr_worker.failed.connect(self.scoreboard_ocr_failed)
        self._ocr_worker.done.connect(self._ocr_thread.quit)
        self._ocr_worker.failed.connect(self._ocr_thread.quit)
        self._ocr_thread.finished.connect(self._ocr_thread.deleteLater)
        self._ocr_active = True
        self._ocr_thread.start()

    def show_ocr_progress(self, percent: float, detail: str):
        theme.set_tone(self.scoreboard_ocr_status, "info")
        self.scoreboard_ocr_status.setText(f"Reading scoreboard frames — {percent:.0f}% — {detail}")

    def show_scoreboard_ocr(self, result: dict | None):
        if not result:
            theme.set_tone(self.scoreboard_ocr_status, "info")
            self.scoreboard_ocr_status.setText("Seek to the final results screen, then draw tight red and blue score ROIs in Analysis. OCR proposes a multi-frame reading for your review.")
            self.use_ocr_btn.setEnabled(False)
            return
        # An OCR reading is a proposal: it keeps the pending tone for the reviewer to check.
        theme.set_tone(self.scoreboard_ocr_status, "pending")
        red, blue = result.get("red", {}), result.get("blue", {})
        total_frames = result.get("frames_sampled", 0)
        def describe(alliance, value):
            score = value.get("value")
            if score is None: return f"{alliance}: no readable score"
            note = "; later OCR read lower—verify the video" if value.get("later_lower_reading") else ""
            if value.get("uncorroborated_prior_high"):
                note += f"; higher reading {max(value.get('outvoted_values') or [score])} was outvoted—verify the video"
            if value.get("flagged"):
                note += "; low confidence—check the final score on the video"
            return f"{alliance}: {score} (evidence {value.get('confidence', 0):.0%}; OCR recognition {value.get('recognition_confidence', 0):.0%}; agreement {value.get('agreement', 0):.0%}; {value.get('samples', 0)}/{total_frames} readable frames{note})"
        self.scoreboard_ocr_status.setText(
            "OCR proposal from final scoreboard window — " + describe("Red", red) + "; " + describe("Blue", blue) +
            ". These are video readings; review them before using the score comparison.")
        self.use_ocr_btn.setEnabled(red.get("value") is not None and blue.get("value") is not None)

    def scoreboard_ocr_done(self, result: dict):
        self.scoreboard_ocr = result
        if self.db and self.video_id is not None:
            self.db.set_scoreboard_ocr(self.video_id, result)
        self.read_scoreboard_btn.setEnabled(True)
        self.show_scoreboard_ocr(result)
        self._ocr_active = False

    def scoreboard_ocr_failed(self, message: str):
        self.read_scoreboard_btn.setEnabled(True)
        theme.set_tone(self.scoreboard_ocr_status, "danger")
        self.scoreboard_ocr_status.setText(f"Scoreboard OCR did not finish: {message}")
        self._ocr_active = False

    def start_live_score(self):
        if not self._results_current(): return
        if not self.db or not self.video_path or self.video_id is None:
            QMessageBox.information(self, "Load a video", "Import or load a video before reading its live score overlay."); return
        self.read_live_score_btn.setEnabled(False)
        self._live_video_id = self.video_id
        self._live_worker = LiveScoreWorker(self.video_path)
        self._live_thread = QThread(self)
        self._live_worker.moveToThread(self._live_thread)
        self._live_thread.started.connect(self._live_worker.run)
        self._live_worker.progress.connect(self.show_live_score_progress)
        self._live_worker.done.connect(self.live_score_done)
        self._live_worker.failed.connect(self.live_score_failed)
        self._live_worker.done.connect(self._live_thread.quit)
        self._live_worker.failed.connect(self._live_thread.quit)
        self._live_thread.finished.connect(self._live_thread.deleteLater)
        self._live_active = True
        self._live_thread.start()

    def show_live_score_progress(self, percent: float, detail: str):
        theme.set_tone(self.live_score_status, "info")
        self.live_score_status.setText(f"Reading the live score overlay — {percent:.0f}% — {detail}")

    def live_score_done(self, result: dict):
        self._live_active = False
        self.read_live_score_btn.setEnabled(True)
        # Switching videos is blocked while reading, but never write one
        # video's events into another.
        if not self.db or self.video_id is None or self.video_id != self._live_video_id:
            return
        try:
            # A read that stopped early replaces proposals only up to where it read.
            until = None if result.get("complete", True) else result.get("read_until_s")
            counts = self.db.replace_live_overlay_events(self.video_id, result["events"], until)
        except ValueError as exc:
            self.live_score_failed(str(exc)); return
        self.refresh_result_tables()
        events = result["events"]
        tips = sum(1 for event in events if event["event_type"].startswith("hive_tip"))
        fouls = sum(1 for event in events if event["event_type"].endswith("_foul"))
        message = (f"Live overlay read: {len(events)} proposed events ({tips} HIVE TIPS, {fouls} foul credits, "
                   f"{len(events) - tips - fouls} other score changes); {counts['added']} added to Review, "
                   f"{counts['kept']} already reviewed or clipped kept. Check each one against the video.")
        tone = "pending"
        if not result.get("complete", True):
            message += (f" The read stopped early ({result.get('stopped_early')}), so events after "
                        f"{result.get('read_until_s', 0):.0f} s are missing.")
            tone = "danger"
        theme.set_tone(self.live_score_status, tone)
        self.live_score_status.setText(message)

    def live_score_failed(self, message: str):
        self._live_active = False
        self.read_live_score_btn.setEnabled(True)
        theme.set_tone(self.live_score_status, "danger")
        self.live_score_status.setText(f"Reading the live score overlay did not finish: {message}")

    def apply_scoreboard_ocr(self):
        if not self.scoreboard_ocr:
            return
        red = self.scoreboard_ocr.get("red", {}).get("value")
        blue = self.scoreboard_ocr.get("blue", {}).get("value")
        if red is None or blue is None:
            return
        self.official_red.setText(str(red))
        self.official_blue.setText(str(blue))
        self.save_official_scores()

    def preview_review_event(self, row: int, column: int):
        item = self.review_table.item(row, 0)
        if not item: return
        event_id = item.data(Qt.ItemDataRole.UserRole)
        clip = next((c for c in self.db.clips_for_video(self.video_id) if c["scoring_event_id"] == event_id), None)
        if clip and Path(clip["path"]).is_file():
            PreviewDialog(Path(clip["path"]), self).exec()
        else:
            event = next((e for e in self.events if e["id"] == event_id), None)
            if event: self.seek(round(event["t_s"] / max(self.duration, 0.001) * 1000))

    def generate_clips(self):
        if not self._results_current(): return
        if not self.db or not self.video_path or not self.events:
            QMessageBox.information(self, "No events", "Process the video and review candidate events first."); return
        ffmpeg = self.ffmpeg_path_edit.text().strip() or None
        folder = self.project_dir() / "clips"; folder.mkdir(exist_ok=True)
        existing = {row["scoring_event_id"] for row in self.db.connection.execute("SELECT c.scoring_event_id FROM clips c JOIN scoring_events e ON e.id=c.scoring_event_id WHERE e.video_id=?", (self.video_id,))}
        made, errors = 0, []
        for event in self.events:
            if event["status"] == "rejected" or event["id"] in existing: continue
            path = folder / f"event-{event['id']}.mp4"
            try:
                start, end = create_highlight(self.video_path, path, event["t_s"], ffmpeg, duration_s=self.duration)
                self.db.add_clip(event["id"], path, start, end); made += 1
            except Exception as exc: errors.append(str(exc))
        self.refresh_clips()
        if errors: QMessageBox.warning(self, "Some clips failed", "\n".join(errors[:5]))
        else: QMessageBox.information(self, "Highlights ready", f"Created {made} highlight clip(s).")

    def refresh_clips(self):
        self.clip_list.clear()
        if not self.db or self.video_id is None: return
        for clip in self.db.clips_for_video(self.video_id):
            item = QListWidgetItem(f"{clip['t_s']:.2f}s — Team {clip['team_number']} — {Path(clip['path']).name}")
            item.setData(Qt.ItemDataRole.UserRole, clip["path"]); self.clip_list.addItem(item)
            frame = thumbnail(Path(clip["path"]))
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, channels = rgb.shape
                image = QImage(rgb.data, w, h, channels * w, QImage.Format.Format_RGB888).copy()
                item.setIcon(QIcon(QPixmap.fromImage(image).scaled(120, 68, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))

    def preview_selected_clip(self, *_):
        item = self.clip_list.currentItem()
        if not item: return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if path.is_file(): PreviewDialog(path, self).exec()
        else: QMessageBox.warning(self, "Clip unavailable", f"The clip file is missing: {path}")

    def export_selected_clip(self):
        if not self._results_current(): return
        item = self.clip_list.currentItem()
        if not item: return
        source = Path(item.data(Qt.ItemDataRole.UserRole))
        target, _ = QFileDialog.getSaveFileName(self, "Export highlight clip", source.name, "MP4 video (*.mp4)")
        if target:
            shutil.copy2(source, target)

    def export_csv(self):
        if not self._results_current(): return
        if not self.db or self.video_id is None: return
        target, _ = QFileDialog.getSaveFileName(self, "Export statistics", "control-vision-stats.csv", "CSV (*.csv)")
        if not target: return
        video = dict(self.db.connection.execute("SELECT * FROM videos WHERE id=?", (self.video_id,)).fetchone())
        export_stats_csv(Path(target), self.db.robots_for_video(self.video_id),
                         [tuple(row) for row in self.db.tracks_for_video(self.video_id)],
                         [dict(row) for row in self.db.cycles_for_video(self.video_id)], self.zones,
                         events=[dict(row) for row in self.db.events_for_video(self.video_id)], video=video)
        QMessageBox.information(self, "Export complete", f"Saved {target}")

    def export_json(self):
        if not self._results_current(): return
        if not self.db or self.video_id is None: return
        target, _ = QFileDialog.getSaveFileName(self, "Export Scout JSON", "control-vision-match.json", "JSON (*.json)")
        if not target: return
        video = dict(self.db.connection.execute("SELECT * FROM videos WHERE id=?", (self.video_id,)).fetchone())
        export_scout_json(Path(target), video, self.db.robots_for_video(self.video_id),
                          [tuple(row) for row in self.db.tracks_for_video(self.video_id)],
                          [dict(row) for row in self.db.cycles_for_video(self.video_id)],
                          [dict(row) for row in self.db.events_for_video(self.video_id)],
                          [dict(row) for row in self.db.clips_for_video(self.video_id)],
                          [dict(row) for row in self.db.photos_for_video(self.video_id)], self.zones,
                          {"red": int(self.official_red.text()) if self.official_red.text().isdigit() else None,
                           "blue": int(self.official_blue.text()) if self.official_blue.text().isdigit() else None})
        QMessageBox.information(self, "Export complete", f"Saved {target}")

    def export_photos(self):
        if not self._results_current(): return
        if not self.db or self.video_id is None: return
        folder = QFileDialog.getExistingDirectory(self, "Choose robot photo export folder")
        if not folder: return
        rows = self.db.photos_for_video(self.video_id); copied = 0
        for row in rows:
            src = Path(row["path"])
            if src.is_file():
                shutil.copy2(src, Path(folder) / f"team-{row['team_number']}.png"); copied += 1
        QMessageBox.information(self, "Photos exported", f"Exported {copied} robot photo(s). Photos below 200×200 pixels are skipped and listed in the run log.")

    def processing_failed(self, message):
        try:
            if self.db and self.video_id is not None:
                self.db.set_analysis_status(self.video_id, "failed")
        except Exception as exc:
            message += f" The failed status could not be saved either: {exc}"
        self._set_results_enabled(False)
        self._show_status(message, "danger")
        self.log_view.addItem(message)
        self.process_btn.setEnabled(True)
        self._processing_active = False
        self.status_progress.hide()
        self._refresh_workflow()

    def cancel_process(self):
        if self._worker: self._worker.cancel(); self.statusBar().showMessage("Cancelling after the current frame…")

    def seek(self, value):
        self.current_time = self.duration * value / 1000 if self.duration else 0
        if self.capture and (self._thread is None or not self._thread.isRunning()):
            self.capture.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
            self.refresh_video()

    def refresh_video(self):
        if not self.capture: return
        if self.timer.isActive(): self.current_time = min(self.duration, self.current_time + 0.1)
        self.capture.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
        ok, frame = self.capture.read()
        if ok:
            self.canvas.show_frame(frame)
            self.field_map.current_time = self.current_time
            self.field_map.update()
            if self.duration: self.slider.blockSignals(True); self.slider.setValue(round(1000 * self.current_time / self.duration)); self.slider.blockSignals(False)
            self.time_label.setText(f"{self.current_time:05.1f}s / {self.duration:05.1f}s")
        if self.current_time >= self.duration: self.timer.stop()

    def closeEvent(self, event):
        for worker, thread, done_slot, failed_slot in (
            (self._worker, self._thread, self.processing_done, self.processing_failed),
            (self._ocr_worker, self._ocr_thread, self.scoreboard_ocr_done, self.scoreboard_ocr_failed),
            (self._live_worker, self._live_thread, self.live_score_done, self.live_score_failed),
        ):
            if worker:
                worker.cancel()
                for signal, slot in ((worker.done, done_slot), (worker.failed, failed_slot)):
                    try:
                        signal.disconnect(slot)
                    except (RuntimeError, TypeError):
                        pass
            try:
                running = thread is not None and thread.isRunning()
            except RuntimeError:  # Qt already deleted a finished thread.
                running = False
            if running:
                thread.quit()
                if not thread.wait(30000):
                    # The window stays open, so the worker's completion must
                    # still clear its busy state when it finally finishes.
                    if worker:
                        worker.done.connect(done_slot)
                        worker.failed.connect(failed_slot)
                    self.statusBar().showMessage("Still cancelling processing; close again after it finishes.")
                    event.ignore()
                    return
        if self.capture: self.capture.release()
        if self.db: self.db.close()
        event.accept()


def run():
    app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec())
