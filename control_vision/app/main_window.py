from __future__ import annotations

import logging
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSpinBox, QStatusBar, QVBoxLayout, QWidget,
)

from control_vision.pipeline.ingest import VideoError, probe_video
from control_vision.pipeline.tracking import track_video
from control_vision.storage.database import ProjectDatabase


class FrameCanvas(QLabel):
    clicked = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111827;color:#cbd5e1;border:1px solid #334155")
        self._frame_size = (0, 0)

    def show_frame(self, frame: np.ndarray, annotations: list[tuple[int, int, str]] | None = None):
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


class FieldMap(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 360)
        self.tracks: dict[int, list[tuple[float, float, float]]] = {}
        self.current_time = 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(18, 18, -18, -18)
        painter.fillRect(rect, QColor("#14532d"))
        painter.setPen(QPen(QColor("#bbf7d0"), 2))
        painter.drawRect(rect)
        for i in range(1, 3):
            x, y = rect.left() + rect.width() * i / 3, rect.top() + rect.height() * i / 3
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
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
            self.done.emit(track_video(*self.args[:5], cancel=lambda: self.cancelled,
                                       progress=report, logger=self.args[5]))
        except InterruptedError:
            self.failed.emit("Processing was cancelled.")
        except Exception as exc:
            self.args[-1].exception("Processing failed")
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Vision — Phase 1 robot tracking")
        self.resize(1180, 800)
        self.db: ProjectDatabase | None = None
        self.project_id = None
        self.video_id = None
        self.video_path: Path | None = None
        self.metadata = None
        self.corners: list[list[float]] = []
        self.seed_points: list[tuple[int, int, int]] = []
        self.robot_labels: list[str] = []
        self.tracks: dict[int, list[tuple[float, float, float]]] = {}
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
        self.process_btn = QPushButton("Process")
        self.cancel_btn = QPushButton("Cancel")
        self.sample_rate = QComboBox(); self.sample_rate.addItems(["5", "6", "7", "8", "9", "10"]); self.sample_rate.setCurrentText("8")
        for button in (self.project_btn, self.open_project_btn, self.video_btn, self.load_video_btn, self.corner_btn, self.robot_btn, self.process_btn, self.cancel_btn):
            toolbar.addWidget(button)
        toolbar.addWidget(QLabel("Sample fps")); toolbar.addWidget(self.sample_rate)
        layout.addLayout(toolbar)

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
        self.setCentralWidget(root)

        self.project_btn.clicked.connect(self.new_project)
        self.open_project_btn.clicked.connect(self.open_project)
        self.video_btn.clicked.connect(self.import_video)
        self.load_video_btn.clicked.connect(self.load_saved_video)
        self.corner_btn.clicked.connect(self.select_corner)
        self.robot_btn.clicked.connect(self.seed_robot)
        self.process_btn.clicked.connect(self.process)
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.canvas.clicked.connect(self.canvas_click)
        self.slider.valueChanged.connect(self.seek)
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.refresh_video)
        self._mode = None
        self._worker = None
        self._thread = None

    def new_project(self):
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
            (Path(folder) / "run_logs").mkdir(exist_ok=True)
            self.statusBar().showMessage(f"Project ready: {name.strip()}")
        except Exception as exc:
            QMessageBox.critical(self, "Could not create project", str(exc))

    def open_project(self):
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
            self.statusBar().showMessage(f"Project opened: {project['name']}")
        except Exception as exc:
            self.db = None
            QMessageBox.critical(self, "Could not open project", str(exc))

    def import_video(self):
        if not self.db or self.project_id is None:
            QMessageBox.information(self, "Create a project", "Create or open a project before importing a video.")
            return
        selected, _ = QFileDialog.getOpenFileName(self, "Import match video", "", "Videos (*.mp4 *.mov *.mkv)")
        if not selected:
            return
        path = Path(selected)
        try:
            metadata = probe_video(path)
            video_id = self.db.add_video(self.project_id, path, metadata)
            cap = cv2.VideoCapture(str(path)); ok, frame = cap.read(); cap.release()
            if not ok:
                raise VideoError(f"Could not read the first frame from {path.name}.")
        except Exception as exc:
            QMessageBox.critical(self, "Video import failed", str(exc)); return
        if self.capture: self.capture.release()
        self.video_id, self.video_path, self.metadata = video_id, path, metadata
        self.corners.clear(); self.seed_points.clear(); self.robot_labels.clear(); self.tracks.clear()
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

    def load_saved_video(self):
        if not self.db or self.project_id is None:
            QMessageBox.information(self, "Open a project", "Open a project before loading one of its saved videos.")
            return
        videos = self.db.connection.execute(
            "SELECT id,path,duration_s,width,height FROM videos WHERE project_id=? ORDER BY id DESC", (self.project_id,)
        ).fetchall()
        if not videos:
            QMessageBox.information(self, "No videos", "This project has no imported videos yet.")
            return
        choices = [f"#{row['id']} {Path(row['path']).name} — {row['duration_s'] or 0:.1f}s" for row in videos]
        from PySide6.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(self, "Load saved video", "Video", choices, 0, False)
        if not ok: return
        video = videos[choices.index(choice)]
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
        calibration = self.db.connection.execute("SELECT corners_json FROM calibrations WHERE video_id=?", (self.video_id,)).fetchone()
        self.corners = json.loads(calibration["corners_json"]) if calibration else []
        robots = self.db.connection.execute("SELECT id,team_number,alliance,seed_x,seed_y FROM robots WHERE video_id=? ORDER BY id", (self.video_id,)).fetchall()
        self.seed_points = [(row["id"], row["seed_x"], row["seed_y"]) for row in robots if row["seed_x"] is not None and row["seed_y"] is not None]
        self.robot_labels = [f"{row['team_number']} ({row['alliance']})" for row in robots]
        self.canvas.show_frame(frame, [(int(x), int(y), str(index + 1)) for index, (x, y) in enumerate(self.corners)])
        self.progress_label.setText(f"Loaded {path.name}. Corners: {len(self.corners)}/4; seeded identities: {len(self.seed_points)}.")
        rows = self.db.connection.execute("SELECT robot_id,t_s,x_ft,y_ft FROM tracks WHERE robot_id IN (SELECT id FROM robots WHERE video_id=?) ORDER BY frame", (self.video_id,)).fetchall()
        track_by_id = {}
        for row in rows: track_by_id.setdefault(row["robot_id"], []).append((row["t_s"], row["x_ft"], row["y_ft"]))
        self.tracks = {i: values for i, values in enumerate(track_by_id.values())}
        self.field_map.tracks = self.tracks; self.field_map.current_time = 0; self.field_map.update()

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
                    self.db.save_calibration(self.video_id, self.corners, {})
                    self.progress_label.setText(f"Calibration saved. Four-corner fit residual: {reprojection_error_px(self.corners):.2f} px (geometric validity requires holdout evaluation).")
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

    def process(self):
        if not self.db or not self.video_path or len(self.corners) != 4 or not self.seed_points:
            QMessageBox.information(self, "Setup incomplete", "Create a project, import a video, select four corners, and seed at least one robot first.")
            return
        model_path, _ = QFileDialog.getOpenFileName(self, "Select local YOLO11n ONNX model", "models", "ONNX model (*.onnx)")
        if not model_path: return
        log_dir = Path(self.db.path).parent / "run_logs"
        log_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logger = logging.getLogger(f"control_vision.{stamp}"); self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir / f"run-{stamp}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); self.logger.addHandler(handler)
        self.logger.info("Video probed: %s; %s", self.video_path, self.metadata)
        self.process_btn.setEnabled(False)
        self._worker = ProcessingWorker((self.video_path, Path(model_path), self.corners.copy(), self.seed_points.copy(), float(self.sample_rate.currentText()), self.logger))
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
        self.db.replace_tracks(rows, [seed[0] for seed in self.seed_points])
        by_id = {}
        for robot_id, frame, t_s, x, y, conf in rows:
            by_id.setdefault(robot_id, []).append((t_s, x, y))
        self.tracks = {i: points for i, points in enumerate(by_id.values())}
        self.field_map.tracks = self.tracks; self.field_map.update()
        self.log_view.addItem(f"Tracking complete: {len(rows):,} mapped robot samples")
        self.progress_label.setText(f"Phase 1 processing complete. Mapped samples: {len(rows):,}. Acceptance targets still require holdout measurement.")
        self.process_btn.setEnabled(True)
        if self.capture: self.timer.start()

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
        if self._worker: self._worker.cancel()
        if self.capture: self.capture.release()
        if self.db: self.db.close()
        event.accept()


def run():
    app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec())
