from __future__ import annotations

import logging
import json
import csv
import shutil
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF, QIcon
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSpinBox, QStatusBar, QVBoxLayout, QWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QDoubleSpinBox, QInputDialog, QHeaderView, QListWidgetItem, QCheckBox, QTextBrowser,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

from control_vision.pipeline.ingest import VideoError, probe_video
from control_vision.pipeline.tracking import track_video
from control_vision.pipeline.manual_tracking import track_seeded_video
from control_vision.pipeline.cycles import count_zone_cycles
from control_vision.pipeline.clips import create_highlight, thumbnail
from control_vision.pipeline.stats import robot_statistics
from control_vision.pipeline.scoreboard_ocr import read_scoreboard
from control_vision.scout.exports import export_scout_json, export_stats_csv
from control_vision.storage.database import ProjectDatabase
from control_vision.game.scoreboard import calculate_scoreboard, classify_match_time


class FrameCanvas(QLabel):
    clicked = Signal(int, int)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
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
        self.setMinimumSize(360, 360)
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
                                     progress=report, logger=self.args[5], photos_dir=self.args[6], confidence_threshold=self.args[7], goal_roi=self.args[8], reid_model_path=self.args[9])
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


def _video_switch_blocked(window) -> bool:
    for thread in (getattr(window, "_thread", None), getattr(window, "_ocr_thread", None)):
        try:
            if thread is not None and thread.isRunning():
                QMessageBox.information(window, "Analysis in progress",
                                        "Wait for processing or scoreboard OCR to finish, or cancel processing, before changing projects or videos.")
                return True
        except RuntimeError:
            # Qt may already have deleted a finished OCR thread.
            continue
    return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Vision — Match analysis")
        self.resize(1180, 800)
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
        self.setStatusBar(QStatusBar())

        root = QWidget(); layout = QVBoxLayout(root)
        toolbar = QHBoxLayout()
        self.project_btn = QPushButton("New Project")
        self.open_project_btn = QPushButton("Open Project")
        self.video_btn = QPushButton("Import Video")
        self.load_video_btn = QPushButton("Load Saved Video")
        self.corner_btn = QPushButton("Calibrate: select 4 corners")
        self.robot_btn = QPushButton("Seed robots")
        self.reuse_calibration_btn = QPushButton("Reuse Calibration")
        self.process_btn = QPushButton("Process")
        self.cancel_btn = QPushButton("Cancel")
        self.sample_rate = QComboBox(); self.sample_rate.addItems(["5", "6", "7", "8", "9", "10"]); self.sample_rate.setCurrentText("8")
        for button in (self.project_btn, self.open_project_btn, self.video_btn, self.load_video_btn, self.corner_btn, self.robot_btn, self.process_btn, self.cancel_btn):
            toolbar.addWidget(button)
        toolbar.addWidget(QLabel("Sample fps")); toolbar.addWidget(self.sample_rate)
        layout.addLayout(toolbar)
        zone_toolbar = QHBoxLayout()
        self.zone_type = QComboBox(); self.zone_type.addItems(["scoring", "pickup", "park"])
        self.zone_btn = QPushButton("Draw Zone")
        self.goal_roi_btn = QPushButton("Draw scoring-element ROI")
        self.red_score_roi_btn = QPushButton("Set red scoreboard ROI")
        self.blue_score_roi_btn = QPushButton("Set blue scoreboard ROI")
        self.zone_remove_combo = QComboBox(); self.zone_remove_btn = QPushButton("Remove selected zone")
        self.zone_help = QLabel("Draw zones on the map; double-click to finish a polygon. Geometry is user supplied.")
        zone_toolbar.addWidget(self.reuse_calibration_btn); zone_toolbar.addWidget(self.zone_type); zone_toolbar.addWidget(self.zone_btn); zone_toolbar.addWidget(self.goal_roi_btn); zone_toolbar.addWidget(self.red_score_roi_btn); zone_toolbar.addWidget(self.blue_score_roi_btn); zone_toolbar.addWidget(self.zone_remove_combo); zone_toolbar.addWidget(self.zone_remove_btn); zone_toolbar.addWidget(self.zone_help, 1)
        layout.addLayout(zone_toolbar)

        body = QHBoxLayout()
        self.canvas = FrameCanvas(); self.field_map = FieldMap()
        body.addWidget(self.canvas, 3); body.addWidget(self.field_map, 2)
        layout.addLayout(body, 1)
        bottom = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, 1000)
        self.time_label = QLabel("00:00 / 00:00")
        bottom.addWidget(self.slider, 1); bottom.addWidget(self.time_label)
        layout.addLayout(bottom)
        self.progress_label = QLabel("Create a project, import a video, calibrate and seed robot identities to begin.")
        layout.addWidget(self.progress_label)
        self.log_view = QListWidget(); self.log_view.setMaximumHeight(100)
        layout.addWidget(self.log_view)
        self.tabs = QTabWidget()
        home_page = QWidget(); home_layout = QVBoxLayout(home_page)
        self.home_project_label = QLabel("No project is open")
        self.home_project_label.setStyleSheet("font-size:20px;font-weight:600")
        self.home_recent = QListWidget()
        self.home_recent.itemDoubleClicked.connect(lambda item: self.load_saved_video_id(item.data(Qt.ItemDataRole.UserRole)))
        home_actions = QHBoxLayout()
        home_new = QPushButton("New Project"); home_open = QPushButton("Open Project")
        home_import = QPushButton("Import Video"); home_load = QPushButton("Load Selected Video")
        home_new.clicked.connect(self.new_project); home_open.clicked.connect(self.open_project)
        home_import.clicked.connect(self.import_video)
        home_load.clicked.connect(lambda: self.load_saved_video_id(self.home_recent.currentItem().data(Qt.ItemDataRole.UserRole) if self.home_recent.currentItem() else None))
        for button in (home_new, home_open, home_import, home_load): home_actions.addWidget(button)
        home_layout.addWidget(self.home_project_label); home_layout.addLayout(home_actions)
        home_layout.addWidget(QLabel("Recent videos in this project — double-click to load")); home_layout.addWidget(self.home_recent)
        self.tabs.addTab(home_page, "Home / Projects")
        self.tabs.addTab(root, "Analysis")
        self.cycles_table = QTableWidget(0, 6)
        self.cycles_table.setHorizontalHeaderLabels(["Team", "Alliance", "Start", "End", "Pickup", "Scoring"])
        self.cycles_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cycles_table.cellDoubleClicked.connect(self.jump_to_cycle)
        cycles_page = QWidget(); cycles_layout = QVBoxLayout(cycles_page)
        self.stats_table = QTableWidget(0, 9)
        self.stats_table.setHorizontalHeaderLabels(["Team", "Alliance", "Cycles", "Avg cycle (s)", "Top speed (ft/s)", "Distance (ft)", "Time per zone (s)", "Confirmed points", "Ranking points"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        cycles_layout.addWidget(QLabel("Zone-trip cycle list. Cycle counts use pickup-to-scoring zone transitions and have medium confidence until ball detection is available."))
        cycles_layout.addWidget(self.cycles_table); cycles_layout.addWidget(QLabel("Per-robot movement statistics")); cycles_layout.addWidget(self.stats_table)
        self.tabs.addTab(cycles_page, "Cycles & Stats")
        self.review_table = QTableWidget(0, 9)
        self.review_table.setHorizontalHeaderLabels(["Time", "Team", "Robot alliance", "Points to", "Event type", "Points", "Confidence", "Status", "Clip"])
        self.review_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.review_table.cellDoubleClicked.connect(self.preview_review_event)
        review_page = QWidget(); review_layout = QVBoxLayout(review_page)
        review_layout.addWidget(QLabel("Zone crossings are candidates only. Enter points after reviewing the video clip; unconfirmed events do not count toward the video total."))
        review_layout.addWidget(self.review_table)
        scoring_config_path = Path(__file__).resolve().parents[1] / "config" / "scoring_biobuzz.json"
        self.scoring_config = json.loads(scoring_config_path.read_text(encoding="utf-8"))
        self.event_types = {entry["id"]: entry for entry in self.scoring_config["review_event_types"]}
        review_controls = QHBoxLayout(); self.review_robot = QComboBox(); self.review_points = QSpinBox(); self.review_points.setRange(-999, 999)
        self.review_alliance = QComboBox(); self.review_alliance.addItem("Robot alliance", None); self.review_alliance.addItem("Red", "red"); self.review_alliance.addItem("Blue", "blue")
        self.review_confidence = QComboBox(); self.review_confidence.addItems(["low", "medium", "high"])
        self.review_type = QComboBox(); self.review_type.addItem("Unclassified / other", "unclassified")
        for event_type in self.scoring_config["review_event_types"]: self.review_type.addItem(event_type["label"], event_type["id"])
        self.confirm_btn = QPushButton("Confirm"); self.reject_btn = QPushButton("Reject")
        review_controls.addWidget(QLabel("Event")); review_controls.addWidget(self.review_type); review_controls.addWidget(QLabel("Attributed robot")); review_controls.addWidget(self.review_robot); review_controls.addWidget(QLabel("Points")); review_controls.addWidget(self.review_points); review_controls.addWidget(QLabel("Points go to")); review_controls.addWidget(self.review_alliance); review_controls.addWidget(QLabel("Confidence")); review_controls.addWidget(self.review_confidence); review_controls.addWidget(self.confirm_btn); review_controls.addWidget(self.reject_btn)
        review_layout.addLayout(review_controls)
        self.review_rule_hint = QLabel("Scoring is awarded to an alliance under the manual; select robot attribution and point recipient separately."); self.review_rule_hint.setWordWrap(True); review_layout.addWidget(self.review_rule_hint)
        self.tabs.addTab(review_page, "Review Queue")
        self.scoring_table = QTableWidget(0, 8)
        self.scoring_table.setHorizontalHeaderLabels(["Time", "Team", "Robot alliance", "Points to", "Event type", "Points", "Confidence", "Status"])
        self.scoring_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scoring_table.cellDoubleClicked.connect(self.jump_to_event)
        scoring_page = QWidget(); scoring_layout = QVBoxLayout(scoring_page)
        self.add_event_btn = QPushButton("Add scoring event at current video time")
        scoring_layout.addWidget(self.add_event_btn); scoring_layout.addWidget(self.scoring_table)
        self.tabs.addTab(scoring_page, "Scoring Timeline")
        self.clip_list = QListWidget(); self.clip_list.itemDoubleClicked.connect(self.preview_selected_clip)
        clips_page = QWidget(); clips_layout = QVBoxLayout(clips_page)
        self.generate_clips_btn = QPushButton("Generate highlight clips for scoring candidates")
        self.preview_clip_btn = QPushButton("Preview selected clip")
        self.clip_export_btn = QPushButton("Export selected clip")
        clips_controls = QHBoxLayout(); clips_controls.addWidget(self.generate_clips_btn); clips_controls.addWidget(self.preview_clip_btn); clips_controls.addWidget(self.clip_export_btn)
        clips_layout.addLayout(clips_controls); clips_layout.addWidget(self.clip_list); self.tabs.addTab(clips_page, "Highlights")
        export_page = QWidget(); export_layout = QVBoxLayout(export_page)
        self.official_red = QLineEdit(); self.official_blue = QLineEdit(); self.official_red.setPlaceholderText("Optional"); self.official_blue.setPlaceholderText("Optional")
        official_row = QHBoxLayout(); official_row.addWidget(QLabel("Scoreboard / official red total")); official_row.addWidget(self.official_red); official_row.addWidget(QLabel("Scoreboard / official blue total")); official_row.addWidget(self.official_blue)
        export_layout.addLayout(official_row)
        self.read_scoreboard_btn = QPushButton("Read final scoreboard from video")
        self.use_ocr_btn = QPushButton("Use OCR reading in score comparison")
        self.use_ocr_btn.setEnabled(False)
        self.scoreboard_ocr_status = QLabel("Scoreboard OCR uses user-drawn red and blue ROIs and a multi-frame vote. Review the proposal before using it.")
        self.scoreboard_ocr_status.setWordWrap(True)
        ocr_row = QHBoxLayout(); ocr_row.addWidget(self.read_scoreboard_btn); ocr_row.addWidget(self.use_ocr_btn)
        export_layout.addLayout(ocr_row); export_layout.addWidget(self.scoreboard_ocr_status)
        self.score_compare = QLabel("Video totals include confirmed events only."); export_layout.addWidget(self.score_compare)
        rp_row = QHBoxLayout()
        self.disq_red = QCheckBox("Red disqualified"); self.disq_blue = QCheckBox("Blue disqualified")
        self.swarm_threshold = QSpinBox(); self.swarm_threshold.setRange(0, 999); self.swarm_threshold.setValue(16)
        self.pollinator_1_threshold = QSpinBox(); self.pollinator_1_threshold.setRange(0, 99); self.pollinator_1_threshold.setValue(4)
        self.pollinator_2_threshold = QSpinBox(); self.pollinator_2_threshold.setRange(0, 99); self.pollinator_2_threshold.setValue(7)
        for widget in (self.disq_red, self.disq_blue, QLabel("SWARM threshold"), self.swarm_threshold, QLabel("POLLINATOR 1 tips"), self.pollinator_1_threshold, QLabel("POLLINATOR 2 tips"), self.pollinator_2_threshold): rp_row.addWidget(widget)
        export_layout.addLayout(rp_row)
        self.ranking_summary = QLabel("Ranking Points follow the selected event thresholds."); self.ranking_summary.setWordWrap(True); export_layout.addWidget(self.ranking_summary)
        self.csv_export_btn = QPushButton("Export statistics CSV"); self.json_export_btn = QPushButton("Export Scout JSON")
        self.photos_export_btn = QPushButton("Export robot photos")
        export_row = QHBoxLayout(); export_row.addWidget(self.csv_export_btn); export_row.addWidget(self.json_export_btn); export_row.addWidget(self.photos_export_btn); export_layout.addLayout(export_row)
        self.tabs.addTab(export_page, "Export")
        settings_page = QWidget(); settings_layout = QFormLayout(settings_page)
        self.model_path_edit = QLineEdit(); self.reid_path_edit = QLineEdit(); self.ffmpeg_path_edit = QLineEdit()
        self.tracking_mode = QComboBox(); self.tracking_mode.addItem("YOLO11 local model (recommended with trained weights)", "yolo")
        self.tracking_mode.addItem("Seeded optical flow preview (no model; not accurate enough for scoring)", "seeded_optical_flow")
        self.model_browse_btn = QPushButton("Browse…"); self.reid_browse_btn = QPushButton("Browse…"); self.ffmpeg_browse_btn = QPushButton("Browse…")
        model_row = QHBoxLayout(); model_row.addWidget(self.model_path_edit); model_row.addWidget(self.model_browse_btn)
        reid_row = QHBoxLayout(); reid_row.addWidget(self.reid_path_edit); reid_row.addWidget(self.reid_browse_btn)
        ffmpeg_row = QHBoxLayout(); ffmpeg_row.addWidget(self.ffmpeg_path_edit); ffmpeg_row.addWidget(self.ffmpeg_browse_btn)
        self.confidence_spin = QDoubleSpinBox(); self.confidence_spin.setRange(0.05, 0.95); self.confidence_spin.setSingleStep(0.05); self.confidence_spin.setValue(0.25)
        self.settings_save_btn = QPushButton("Save settings")
        settings_layout.addRow("Tracking method", self.tracking_mode); settings_layout.addRow("YOLO11n ONNX model", model_row); settings_layout.addRow("Optional local ReID embeddings (ONNX) or compatible YOLO .pt", reid_row); settings_layout.addRow("FFmpeg executable (blank uses PATH)", ffmpeg_row); settings_layout.addRow("Detection confidence", self.confidence_spin); settings_layout.addRow("Sample rate (frames per second)", self.sample_rate); settings_layout.addRow(self.settings_save_btn)
        self.tabs.addTab(settings_page, "Settings")
        rules_page = QWidget(); rules_layout = QVBoxLayout(rules_page)
        self.rules_browser = QTextBrowser(); self.rules_browser.setOpenExternalLinks(True)
        self.rules_browser.setMarkdown(self.build_rules_reference())
        rules_layout.addWidget(self.rules_browser); self.tabs.addTab(rules_page, "BIOBUZZ Rules")
        self.setCentralWidget(self.tabs)

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
        self.use_ocr_btn.clicked.connect(self.apply_scoreboard_ocr)
        self.official_red.textChanged.connect(self.update_score_comparison)
        self.official_blue.textChanged.connect(self.update_score_comparison)
        self.review_alliance.currentIndexChanged.connect(self.update_score_comparison)
        self.official_red.editingFinished.connect(self.save_official_scores)
        self.official_blue.editingFinished.connect(self.save_official_scores)
        self.disq_red.clicked.connect(self.save_official_scores); self.disq_blue.clicked.connect(self.save_official_scores)
        self.swarm_threshold.editingFinished.connect(self.save_official_scores); self.pollinator_1_threshold.editingFinished.connect(self.save_official_scores); self.pollinator_2_threshold.editingFinished.connect(self.save_official_scores)
        self.disq_red.stateChanged.connect(self.update_score_comparison); self.disq_blue.stateChanged.connect(self.update_score_comparison)
        self.swarm_threshold.valueChanged.connect(self.update_score_comparison); self.pollinator_1_threshold.valueChanged.connect(self.update_score_comparison); self.pollinator_2_threshold.valueChanged.connect(self.update_score_comparison)
        self.settings_save_btn.clicked.connect(self.save_settings)
        self.model_browse_btn.clicked.connect(self.browse_model)
        self.reid_browse_btn.clicked.connect(self.browse_reid)
        self.ffmpeg_browse_btn.clicked.connect(self.browse_ffmpeg)
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.refresh_video)
        self._mode = None
        self._worker = None
        self._thread = None
        self._ocr_worker = None
        self._ocr_thread = None
        self.scoreboard_ocr = None

    def project_dir(self) -> Path | None:
        return self.db.path.parent if self.db else None

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
            self.home_project_label.setText("No project is open")
            return
        project = self.db.connection.execute("SELECT name FROM projects WHERE id=?", (self.project_id,)).fetchone()
        self.home_project_label.setText(f"{project['name']} — {self.db.path}")
        for row in self.db.connection.execute("SELECT id,path,event_code,match_number,duration_s FROM videos WHERE project_id=? ORDER BY id DESC LIMIT 20", (self.project_id,)):
            label = f"{Path(row['path']).name}  ·  {row['event_code'] or 'Practice'} {row['match_number'] or ''}  ·  {row['duration_s'] or 0:.0f}s"
            item = QListWidgetItem(label); item.setData(Qt.ItemDataRole.UserRole, int(row['id'])); self.home_recent.addItem(item)

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
        self.statusBar().showMessage(f"Outline the full {alliance} score digits, including their baselines. Click at least three corners, then double-click to save.")

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
        self.zones[clean_name] = {"type": self.zone_type.currentText(), "points": self.field_map.draft.copy()}
        self.zone_remove_combo.addItem(clean_name)
        self.field_map.zones = self.zones; self.field_map.draft = []; self._drawing_zone = False
        self.db.save_calibration(self.video_id, self.corners, self.zones)
        self.field_map.update(); self.statusBar().showMessage(f"Saved zone '{clean_name}'.")

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
            QMessageBox.critical(self, "Could not open project", str(exc))

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
        self.use_ocr_btn.setEnabled(False)
        self.scoreboard_ocr_status.setText("Scoreboard OCR uses user-drawn red and blue ROIs and a multi-frame vote. Review the proposal before using it.")
        self.disq_red.setChecked(False); self.disq_blue.setChecked(False)
        self.swarm_threshold.setValue(16); self.pollinator_1_threshold.setValue(4); self.pollinator_2_threshold.setValue(7)
        self.cycles_table.setRowCount(0)
        self.stats_table.setRowCount(0)
        self.review_table.setRowCount(0)
        self.scoring_table.setRowCount(0)
        self.clip_list.clear()
        self.progress_label.setText("Project opened. Load or import a match video to begin.")
        self.slider.setValue(0)
        self.refresh_result_tables()
        self.refresh_home()

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
            video_warnings = []
            if metadata["width"] < 1280 or metadata["height"] < 720:
                video_warnings.append(f"Resolution is {metadata['width']}×{metadata['height']}; 1280×720 or higher is recommended.")
            if metadata["fps"] < 15:
                video_warnings.append(f"Frame rate is {metadata['fps']:.1f} fps; 15 fps or higher is recommended.")
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
        self.use_ocr_btn.setEnabled(False)
        self.scoreboard_ocr_status.setText("Draw red and blue scoreboard ROIs, then run OCR to propose the final video scoreboard values.")
        self.disq_red.setChecked(False); self.disq_blue.setChecked(False)
        self.swarm_threshold.setValue(int(self.scoring_config["ranking_points"]["thresholds_by_event_type"]["standard"]["swarm_leave_park_points"]))
        self.pollinator_1_threshold.setValue(int(self.scoring_config["ranking_points"]["thresholds_by_event_type"]["standard"]["pollinator_1_hive_tips"]))
        self.pollinator_2_threshold.setValue(int(self.scoring_config["ranking_points"]["thresholds_by_event_type"]["standard"]["pollinator_2_hive_tips"]))
        self.canvas.roi_points = []
        self.canvas.named_rois = {}
        self.field_map.zones = {}; self.field_map.tracks = {}; self.field_map.update()
        self.duration = float(metadata["duration_s"])
        self.capture = cv2.VideoCapture(str(path))
        self.canvas.show_frame(frame)
        self.slider.setValue(0)
        warnings = []
        if metadata["width"] < 1280 or metadata["height"] < 720: warnings.append("below 720p")
        if metadata["fps"] < 15: warnings.append("below 15 fps")
        message = f"{path.name}: {metadata['width']}×{metadata['height']}, {metadata['fps']:.2f} fps, {self.duration:.1f} s"
        if warnings: message += " — warning: " + " and ".join(warnings)
        self.progress_label.setText(message)
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
            "SELECT id,path,duration_s,width,height,official_red,official_blue,red_disqualified,blue_disqualified,swarm_threshold,pollinator_1_threshold,pollinator_2_threshold,scoreboard_ocr_json,tracking_mode FROM videos WHERE project_id=? AND id=?",
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
        self.duration = float(metadata["duration_s"]); self.current_time = 0
        self.official_red.setText(str(video["official_red"]) if video["official_red"] is not None else "")
        self.official_blue.setText(str(video["official_blue"]) if video["official_blue"] is not None else "")
        self.disq_red.setChecked(bool(video["red_disqualified"])); self.disq_blue.setChecked(bool(video["blue_disqualified"]))
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
        self.events = self.db.events_for_video(self.video_id)
        self.photo_rows = self.db.photos_for_video(self.video_id)
        robots = self.db.connection.execute("SELECT id,team_number,alliance,seed_x,seed_y FROM robots WHERE video_id=? ORDER BY id", (self.video_id,)).fetchall()
        self.seed_points = [(row["id"], row["seed_x"], row["seed_y"]) for row in robots if row["seed_x"] is not None and row["seed_y"] is not None]
        self.robot_labels = [f"{row['team_number']} ({row['alliance']})" for row in robots]
        self.canvas.show_frame(frame, [(int(x), int(y), str(index + 1)) for index, (x, y) in enumerate(self.corners)])
        self.refresh_home()
        self.progress_label.setText(f"Loaded {path.name}. Corners: {len(self.corners)}/4; seeded identities: {len(self.seed_points)}.")
        rows = self.db.connection.execute("SELECT robot_id,t_s,x_ft,y_ft FROM tracks WHERE robot_id IN (SELECT id FROM robots WHERE video_id=?) ORDER BY frame", (self.video_id,)).fetchall()
        track_by_id = {}
        for row in rows: track_by_id.setdefault(row["robot_id"], []).append((row["t_s"], row["x_ft"], row["y_ft"]))
        self.tracks = {i: values for i, values in enumerate(track_by_id.values())}
        self.field_map.tracks = self.tracks; self.field_map.current_time = 0; self.field_map.update()
        self.refresh_result_tables()

    def select_corner(self):
        if self.video_path is None: return
        self._mode = "corner"
        self.statusBar().showMessage(f"Click corner {len(self.corners)+1} of 4 in order (top-left, top-right, bottom-right, bottom-left).")

    def seed_robot(self):
        if self.video_path is None: return
        if len(self.seed_points) >= 4:
            QMessageBox.information(self, "Robot limit", "Up to four robots can be seeded for a match."); return
        self._mode = "robot"
        self.statusBar().showMessage("Click a robot near its ground-contact point to assign its team and alliance.")

    def canvas_click(self, x, y):
        if self._mode == "corner":
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
                    self.progress_label.setText(f"Calibration saved. Four-corner interpolation residual: {reprojection_error_px(self.corners):.2f} px (this is not an independent accuracy check).")
                except Exception as exc:
                    QMessageBox.warning(self, "Invalid calibration", str(exc)); self.corners.clear()
        elif self._mode == "robot":
            dialog = IdentityDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                robot_id = self.db.add_robot(self.video_id, dialog.team.value(), dialog.alliance.currentText(), x, y)
                self.seed_points.append((robot_id, x, y))
                self.robot_labels.append(f"{dialog.team.value()} ({dialog.alliance.currentText()})")
                self.progress_label.setText("Robot identities: " + ", ".join(self.robot_labels))
                if self.capture:
                    self.capture.set(cv2.CAP_PROP_POS_MSEC, 0); ok, frame = self.capture.read()
                    if ok: self.canvas.show_frame(frame, [(sx, sy, label) for (_, sx, sy), label in zip(self.seed_points, self.robot_labels)])
            self._mode = None
        elif self._mode == "roi" or self._mode in {"scoreboard_red_roi", "scoreboard_blue_roi"}:
            self.canvas.roi_points.append([x, y])
            if hasattr(self.canvas, "_last_frame"): self.canvas.show_frame(self.canvas._last_frame)

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
        self._thread.start()

    def show_progress(self, stage, percent, detail):
        self.progress_label.setText(f"{stage} — {percent:.0f}% — {detail}")

    def processing_done(self, rows):
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
        self.progress_label.setText(f"Analysis complete. Samples: {len(track_rows):,}; cycle/event candidates: {len(self.events)}. Review points and statuses before reporting scores.")
        self.refresh_result_tables()
        self.process_btn.setEnabled(True)
        if self.capture: self.timer.start()

    def refresh_result_tables(self):
        if not self.db or self.video_id is None: return
        self.cycles = self.db.cycles_for_video(self.video_id)
        self.events = self.db.events_for_video(self.video_id)
        robots = self.db.robots_for_video(self.video_id)
        self.review_robot.clear()
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
        points_by_robot = {}
        robot_alliances = {int(robot["id"]): robot["alliance"] for robot in robots}
        for event in self.events:
            if event["status"] == "confirmed" and event["robot_id"] is not None and event["event_alliance"] == robot_alliances.get(int(event["robot_id"])):
                points_by_robot[event["robot_id"]] = points_by_robot.get(event["robot_id"], 0) + event["points"]
        board = calculate_scoreboard(self.events, robots, self.swarm_threshold.value(), self.pollinator_1_threshold.value(),
                                     self.pollinator_2_threshold.value(), {a for a, cb in (("red", self.disq_red), ("blue", self.disq_blue)) if cb.isChecked()})
        self.stats_table.setRowCount(len(robots))
        for row_index, robot in enumerate(robots):
            values = stats.get(robot["id"], {"cycles": "Preview only" if preview_only else 0,
                                                "cycle_time_avg_s": None if preview_only else 0,
                                                "top_speed_ft_s": None if preview_only else 0,
                                                "distance_ft": None if preview_only else 0})
            zone_seconds = values.get("zone_time_s", {})
            zone_summary = "Unavailable in preview mode" if preview_only else ", ".join(f"{name}: {seconds:.1f}" for name, seconds in zone_seconds.items())
            display = [robot["team_number"], robot["alliance"], values["cycles"], "—" if preview_only else f"{values['cycle_time_avg_s']:.2f}", "—" if preview_only else f"{values['top_speed_ft_s']:.2f}", "—" if preview_only else f"{values['distance_ft']:.2f}", zone_summary, points_by_robot.get(robot["id"], 0), board["team_ranking_points"].get(int(robot["id"]), 0)]
            for col, value in enumerate(display): self.stats_table.setItem(row_index, col, QTableWidgetItem(str(value)))
        flagged = [event for event in self.events if event["status"] == "flagged"]
        self.review_table.setRowCount(len(flagged))
        clips_by_event = {row["scoring_event_id"]: row for row in self.db.clips_for_video(self.video_id)}
        for row_index, event in enumerate(flagged):
            clip = clips_by_event.get(event["id"])
            event_label = self.event_types.get(event["event_type"], {}).get("label", "Unclassified / other")
            phase, remaining = classify_match_time(float(event["t_s"]), self.scoring_config["match_periods_s"])
            values = [f"{event['t_s']:.2f}s {phase} ({remaining:.0f}s left)", event["team_number"], event["alliance"], event["event_alliance"], event_label, event["points"], event["confidence"], event["status"], Path(clip["path"]).name if clip else "Not generated"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value)); self.review_table.setItem(row_index, col, item)
            self.review_table.item(row_index, 0).setData(Qt.ItemDataRole.UserRole, event["id"])
        self.scoring_table.setRowCount(len(self.events))
        for row_index, event in enumerate(self.events):
            event_label = self.event_types.get(event["event_type"], {}).get("label", "Unclassified / other")
            phase, remaining = classify_match_time(float(event["t_s"]), self.scoring_config["match_periods_s"])
            values = [f"{event['t_s']:.2f}s {phase} ({remaining:.0f}s left)", event["team_number"], event["alliance"], event["event_alliance"], event_label, event["points"], event["confidence"], event["status"]]
            for col, value in enumerate(values): self.scoring_table.setItem(row_index, col, QTableWidgetItem(str(value)))
            self.scoring_table.item(row_index, 0).setData(Qt.ItemDataRole.UserRole, event["t_s"])
        self.refresh_clips()
        self.update_score_comparison()

    def jump_to_cycle(self, row: int, column: int):
        item = self.cycles_table.item(row, 0)
        if item: self.seek(round(float(item.data(Qt.ItemDataRole.UserRole)) / max(self.duration, 0.001) * 1000))

    def jump_to_event(self, row: int, column: int):
        item = self.scoring_table.item(row, 0)
        if item: self.seek(round(float(item.data(Qt.ItemDataRole.UserRole)) / max(self.duration, 0.001) * 1000))

    def add_manual_event(self):
        if not self.db or self.video_id is None or self.review_robot.currentData() is None:
            QMessageBox.information(self, "Set up robot identities", "Seed at least one robot before adding a scoring event."); return
        self.db.add_scoring_event(int(self.review_robot.currentData()), self.current_time)
        self.refresh_result_tables()
        self.statusBar().showMessage(f"Added a flagged scoring candidate at {self.current_time:.2f}s. Classify it after reviewing the video.")

    def review_selected(self, status: str):
        row = self.review_table.currentRow()
        if row < 0 or not self.db: return
        event_id = self.review_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        robot_id = self.review_robot.currentData()
        self.db.review_event(event_id, status, self.review_points.value(), robot_id, self.review_type.currentData(), self.review_confidence.currentText(), self.review_alliance.currentData())
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
        robot_index = self.review_robot.findData(event["robot_id"])
        if robot_index >= 0: self.review_robot.setCurrentIndex(robot_index)
        self.review_points.setValue(int(event["points"]))
        self.review_confidence.setCurrentText(event["confidence"])
        alliance_index = self.review_alliance.findData(event["scoring_alliance"])
        self.review_alliance.setCurrentIndex(alliance_index if alliance_index >= 0 else 0)

    def update_score_comparison(self, *_):
        if not self.db or self.video_id is None:
            self.score_compare.setText("Video totals include confirmed events only."); return
        events = self.db.events_for_video(self.video_id)
        robots = [dict(row) for row in self.db.robots_for_video(self.video_id)]
        has_tracks = self.db.connection.execute("SELECT 1 FROM tracks WHERE robot_id IN (SELECT id FROM robots WHERE video_id=?) LIMIT 1", (self.video_id,)).fetchone()
        if not events and not has_tracks:
            self.score_compare.setText("No analysis has run for this video yet."); return
        disqualified = {alliance for alliance, checkbox in (("red", self.disq_red), ("blue", self.disq_blue)) if checkbox.isChecked()}
        board = calculate_scoreboard(events, robots, self.swarm_threshold.value(), self.pollinator_1_threshold.value(),
                                     self.pollinator_2_threshold.value(), disqualified)
        totals = board["alliance_points"]
        flagged_count = sum(event["status"] == "flagged" for event in events)
        state = "UNREVIEWED — final score pending" if flagged_count else "Reviewed" if any(event["status"] == "confirmed" for event in events) else "No scoring events confirmed"
        parts = [f"{state}. Video total — red {totals['red']}, blue {totals['blue']}, match {sum(totals.values())}; flagged events: {flagged_count}."]
        for alliance, edit in (("red", self.official_red), ("blue", self.official_blue)):
            raw = edit.text().strip()
            if raw.isdigit(): parts.append(f"{alliance.capitalize()} official {int(raw)}; difference {totals[alliance] - int(raw):+d}")
        self.score_compare.setText("  ".join(parts))
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
        disqualified = {alliance for alliance, checkbox in (("red", self.disq_red), ("blue", self.disq_blue)) if checkbox.isChecked()}
        self.db.set_official_scores(self.video_id, int(red) if red else None, int(blue) if blue else None,
                                    disqualified, self.swarm_threshold.value(), self.pollinator_1_threshold.value(), self.pollinator_2_threshold.value())
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
        self._ocr_thread.start()

    def show_ocr_progress(self, percent: float, detail: str):
        self.scoreboard_ocr_status.setText(f"Reading scoreboard frames — {percent:.0f}% — {detail}")

    def show_scoreboard_ocr(self, result: dict | None):
        if not result:
            self.scoreboard_ocr_status.setText("Scoreboard OCR uses user-drawn red and blue ROIs and a multi-frame vote. Review the proposal before using it.")
            self.use_ocr_btn.setEnabled(False)
            return
        red, blue = result.get("red", {}), result.get("blue", {})
        total_frames = result.get("frames_sampled", 0)
        def describe(alliance, value):
            score = value.get("value")
            if score is None: return f"{alliance}: no readable score"
            return f"{alliance}: {score} (evidence {value.get('confidence', 0):.0%}; OCR recognition {value.get('recognition_confidence', 0):.0%}; agreement {value.get('agreement', 0):.0%}; {value.get('samples', 0)}/{total_frames} readable frames)"
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

    def scoreboard_ocr_failed(self, message: str):
        self.read_scoreboard_btn.setEnabled(True)
        self.scoreboard_ocr_status.setText(f"Scoreboard OCR did not finish: {message}")

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
                start, end = create_highlight(self.video_path, path, event["t_s"], ffmpeg, after_s=min(3.0, max(0.1, self.duration - event["t_s"])))
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
        item = self.clip_list.currentItem()
        if not item: return
        source = Path(item.data(Qt.ItemDataRole.UserRole))
        target, _ = QFileDialog.getSaveFileName(self, "Export highlight clip", source.name, "MP4 video (*.mp4)")
        if target:
            shutil.copy2(source, target)

    def export_csv(self):
        if not self.db or self.video_id is None: return
        target, _ = QFileDialog.getSaveFileName(self, "Export statistics", "control-vision-stats.csv", "CSV (*.csv)")
        if not target: return
        events = self.db.events_for_video(self.video_id)
        points = {}
        robot_alliances = {int(robot["id"]): robot["alliance"] for robot in self.db.robots_for_video(self.video_id)}
        for event in events:
            if event["status"] == "confirmed" and event["robot_id"] is not None and event["event_alliance"] == robot_alliances.get(int(event["robot_id"])):
                points[event["robot_id"]] = points.get(event["robot_id"], 0) + event["points"]
        video = self.db.connection.execute("SELECT * FROM videos WHERE id=?", (self.video_id,)).fetchone()
        board = calculate_scoreboard(events, self.db.robots_for_video(self.video_id), int(video["swarm_threshold"]),
                                     int(video["pollinator_1_threshold"]), int(video["pollinator_2_threshold"]),
                                     {a for a, col in (("red", "red_disqualified"), ("blue", "blue_disqualified")) if video[col]})
        export_stats_csv(Path(target), self.db.robots_for_video(self.video_id),
                         [tuple(row) for row in self.db.tracks_for_video(self.video_id)],
                         [dict(row) for row in self.db.cycles_for_video(self.video_id)], self.zones, points, board["team_ranking_points"])
        QMessageBox.information(self, "Export complete", f"Saved {target}")

    def export_json(self):
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
        self.progress_label.setText(message)
        self.log_view.addItem(message)
        self.process_btn.setEnabled(True)

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
