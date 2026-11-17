"""Field map and trails: where every robot went, on the calibrated field (guide page 16).

Trails are the stored positions (feet on the 12 x 12 ft field, measured from the first
corner clicked when calibrating). A break of more than a second between samples is drawn
as a gap, never bridged with a straight line. Ball positions are not stored by this build.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget,
)

from control_vision.app import widgets
from control_vision.app.frameview import FrameView, frame_pixmap, tokens
from control_vision.app.media import format_clock, read_frame
from control_vision.app.pages.calibrate import FIELD_FEET, FieldMap, side_view, warped_field
from control_vision.pipeline.zones import DEFAULT_MAX_GAP_S

HEAT_CELLS = 24          # 0.5 ft squares
CLICK_PX = 10
PLAY_STEP_S = 0.5


@dataclass
class RobotTrail:
    robot_id: int
    team: str
    alliance: str
    samples: list[tuple[float, float, float]] = field(default_factory=list)    # (t_s, x_ft, y_ft)
    visible: bool = True
    colour: str = "#FFFFFF"

    def segments(self, until_s: float | None = None) -> list[list[tuple[float, float, float]]]:
        """Runs of samples with no break longer than the gap limit; a break starts a new segment."""
        runs, run, last_t = [], [], None
        for sample in self.samples:
            if until_s is not None and sample[0] > until_s:
                break
            if last_t is not None and sample[0] - last_t > DEFAULT_MAX_GAP_S:
                runs.append(run); run = []
            run.append(sample)
            last_t = sample[0]
        if run:
            runs.append(run)
        return runs

    def distance_ft(self, until_s: float | None = None) -> float:
        return sum(math.dist(a[1:], b[1:]) for run in self.segments(until_s) for a, b in zip(run, run[1:]))

    def gaps(self) -> int:
        return max(0, len(self.segments()) - 1)


def load_trails(db, video_id: int) -> list[RobotTrail]:
    colours = tokens()
    trails, seen = [], {"red": 0, "blue": 0}
    for robot in db.robots_for_video(video_id):
        alliance = robot["alliance"] if robot["alliance"] in ("red", "blue") else "red"
        base = QColor(colours.red if alliance == "red" else colours.blue)
        colour = base if seen[alliance] == 0 else base.lighter(150)
        seen[alliance] += 1
        trails.append(RobotTrail(int(robot["id"]), str(robot["team_number"]), alliance, colour=colour.name()))
    by_id = {trail.robot_id: trail for trail in trails}
    for row in db.tracks_for_video(video_id):
        trail = by_id.get(int(row["robot_id"]))
        if trail is not None:
            trail.samples.append((float(row["t_s"]), float(row["x_ft"]), float(row["y_ft"])))
    for trail in trails:
        trail.samples.sort()
    return trails


def heat_counts(trails: list[RobotTrail], until_s: float | None = None) -> list[list[int]]:
    grid = [[0] * HEAT_CELLS for _ in range(HEAT_CELLS)]
    for trail in trails:
        if not trail.visible:
            continue
        for t, x, y in trail.samples:
            if until_s is not None and t > until_s:
                break
            column = min(HEAT_CELLS - 1, max(0, int(x / FIELD_FEET * HEAT_CELLS)))
            row = min(HEAT_CELLS - 1, max(0, int(y / FIELD_FEET * HEAT_CELLS)))
            grid[row][column] += 1
    return grid


class TrailMap(FieldMap):
    """The calibrated field with trails, a heatmap, zoom and click-to-seek."""

    point_clicked = Signal(int, float)       # robot id, time in s

    def __init__(self):
        super().__init__()
        self.setAccessibleName("Field map with robot trails")
        self.trails: list[RobotTrail] = []
        self.all_zones: dict = {}
        self.time_s: float | None = None
        self.show_trails = True
        self.show_heatmap = False
        self.show_zones = True
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._press: QPointF | None = None
        self._panned = False

    # ----- geometry with zoom -----

    def field_rect(self) -> QRectF:
        base = super().field_rect()
        side = base.width() * self.zoom
        centre = base.center() + self.pan
        return QRectF(centre.x() - side / 2, centre.y() - side / 2, side, side)

    def set_zoom(self, zoom: float) -> None:
        self.zoom = min(4.0, max(1.0, zoom))
        if self.zoom == 1.0:
            self.pan = QPointF(0, 0)
        self.update()

    # ----- painting -----

    def paintEvent(self, event):
        self.zones = self.all_zones if self.show_zones else {}
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self.rect())
        if self.show_heatmap:
            self._paint_heat(painter)
        if self.show_trails:
            for trail in self.trails:
                if trail.visible:
                    self._paint_trail(painter, trail, faded=self.show_heatmap)
        painter.end()

    def _paint_heat(self, painter: QPainter) -> None:
        grid = heat_counts(self.trails, self.time_s)
        peak = max((value for row in grid for value in row), default=0)
        if not peak:
            return
        rect = self.field_rect()
        cell = rect.width() / HEAT_CELLS
        painter.setPen(Qt.PenStyle.NoPen)
        for row, values in enumerate(grid):
            for column, value in enumerate(values):
                if value:
                    share = value / peak     # blue = less time, red = more time
                    colour = QColor.fromHsvF(0.66 * (1 - share), 0.9, 1.0, 0.25 + 0.45 * share)
                    painter.setBrush(colour)
                    painter.drawRect(QRectF(rect.left() + column * cell, rect.top() + row * cell, cell, cell))

    def _paint_trail(self, painter: QPainter, trail: RobotTrail, faded: bool) -> None:
        colour = QColor(trail.colour)
        if faded:
            colour.setAlpha(90)
        runs = trail.segments(self.time_s)
        if not runs:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(colour, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        for run in runs:
            path = QPainterPath(self.to_widget(run[0][1], run[0][2]))
            for _t, x, y in run[1:]:
                path.lineTo(self.to_widget(x, y))
            painter.drawPath(path)
        start = self.to_widget(runs[0][0][1], runs[0][0][2])
        painter.setBrush(colour)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(start, 4, 4)
        last_run = runs[-1]
        end = self.to_widget(last_run[-1][1], last_run[-1][2])
        if len(last_run) > 1:
            before = self.to_widget(last_run[-2][1], last_run[-2][2])
            angle = math.atan2(end.y() - before.y(), end.x() - before.x())
            if end != before:
                head = QPolygonF([end + QPointF(math.cos(angle) * 9, math.sin(angle) * 9),
                                  end + QPointF(math.cos(angle + 2.5) * 8, math.sin(angle + 2.5) * 8),
                                  end + QPointF(math.cos(angle - 2.5) * 8, math.sin(angle - 2.5) * 8)])
                painter.drawPolygon(head)
        painter.setPen(QColor(tokens().text))
        painter.drawText(end + QPointF(8, -8), trail.team)

    # ----- mouse -----

    def nearest_point(self, point: QPointF) -> tuple[int, float] | None:
        """The trail sample nearest a click; on a tie the earliest wins (when a robot that sat still arrived)."""
        best, best_distance = None, CLICK_PX + 0.5
        for trail in self.trails:
            if not trail.visible:
                continue
            for t, x, y in trail.samples:
                if self.time_s is not None and t > self.time_s:
                    break
                distance = (self.to_widget(x, y) - point).manhattanLength()
                if distance < best_distance:
                    best, best_distance = (trail.robot_id, t), distance
        return best

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press, self._panned = event.position(), False

    def mouseMoveEvent(self, event):
        if self._press is not None and self.zoom > 1.0:
            delta = event.position() - self._press
            if delta.manhattanLength() > 3 or self._panned:
                self.pan += delta
                self._press, self._panned = event.position(), True
                self.update()

    def mouseReleaseEvent(self, event):
        if self._press is None:
            return
        panned, self._press = self._panned, None
        if not panned:
            hit = self.nearest_point(event.position())
            if hit is not None:
                self.point_clicked.emit(*hit)

    def wheelEvent(self, event):
        self.set_zoom(self.zoom * (1.25 if event.angleDelta().y() > 0 else 0.8))


class FieldMapPage(QWidget):
    open_calibrate = Signal(int)
    open_queue = Signal()
    new_project = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        self.db = None
        self.project_id = None
        self.video: dict | None = None
        self.duration = 0.0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        self.no_project = widgets.EmptyState("Open a project to see trails", "", "+ New project", self.new_project.emit)
        outer.addWidget(self.no_project, 1)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        outer.addWidget(self.body, 1)

        top = QHBoxLayout()
        match_label = QLabel("&Match")
        self.video_combo = QComboBox(); self.video_combo.setMinimumWidth(320)
        self.video_combo.setAccessibleName("Match")
        match_label.setBuddy(self.video_combo)
        self.video_combo.currentIndexChanged.connect(lambda _i: self.show_video(self.video_combo.currentData()))
        top.addWidget(match_label); top.addWidget(self.video_combo); top.addStretch(1)
        self.warning = widgets.label("", tone="pending", wrap=True)
        top.addWidget(self.warning, 2)
        body.addLayout(top)

        self._empty_action = lambda: None
        self.empty = widgets.EmptyState("", "", "Open", lambda: self._empty_action())
        body.addWidget(self.empty, 1)

        self.content = QWidget()
        content = QHBoxLayout(self.content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        body.addWidget(self.content, 1)
        self.map = TrailMap()
        self.map.point_clicked.connect(self._point_clicked)
        content.addWidget(self.map, 3)

        side = QVBoxLayout()
        side.addWidget(widgets.section_label("Layers"))
        self.layer_boxes = {}
        for key, text, on in (("trails", "&Trails", True), ("heatmap", "&Heatmap", False),
                              ("zones", "&Zones", True), ("picture", "&Field picture", True)):
            box = QCheckBox(text); box.setChecked(on)
            box.toggled.connect(self._layers_changed)
            side.addWidget(box)
            self.layer_boxes[key] = box
        self.balls = QCheckBox("&Ball positions")
        widgets.disable_with_reason(self.balls, False, "Ball positions are not stored by this build")
        side.addWidget(self.balls)
        side.addWidget(widgets.section_label("Robots"))
        self.legend = QVBoxLayout()
        side.addLayout(self.legend)
        self.preview = FrameView("Click a point on a trail to see that moment")
        self.preview.setMinimumSize(240, 135)
        self.preview.setAccessibleName("Video at the selected time")
        side.addWidget(self.preview)
        self.coordinates = widgets.label("Positions are in feet on the 12 x 12 ft field, from the first corner "
                                         "clicked when calibrating. A break of more than "
                                         f"{DEFAULT_MAX_GAP_S:g} s is shown as a gap.", role="muted", wrap=True)
        side.addWidget(self.coordinates)
        side.addStretch(1)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(widgets.button("Zoom in", on_click=lambda: self.map.set_zoom(self.map.zoom * 1.25)))
        zoom_row.addWidget(widgets.button("Reset", on_click=lambda: self.map.set_zoom(1.0)))
        side.addLayout(zoom_row)
        side_widget = QWidget(); side_widget.setLayout(side); side_widget.setFixedWidth(280)
        content.addWidget(side_widget)

        controls = QHBoxLayout()
        self.play_btn = widgets.button("Play", on_click=self.toggle_play)
        controls.addWidget(self.play_btn)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setAccessibleName("Match time")
        self.scrubber.valueChanged.connect(self._scrubbed)
        controls.addWidget(self.scrubber, 1)
        self.time_label = widgets.label("00:00")
        controls.addWidget(self.time_label)
        body.addLayout(controls)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.no_project.show(); self.body.hide()

    # ----- project and match -----

    def set_project(self, db, project_id) -> None:
        same = db is self.db and project_id == self.project_id
        self.db, self.project_id = db, project_id
        has = db is not None and project_id is not None
        self.no_project.setVisible(not has)
        self.body.setVisible(has)
        if not same:
            self.video = None
        self.refresh_videos()

    def refresh_videos(self, select: int | None = None) -> None:
        current = select if select is not None else (self.video["id"] if self.video else None)
        rows = [] if self.db is None else self.db.connection.execute(
            "SELECT id,path,match_number,analysis_status FROM videos WHERE project_id=? ORDER BY id",
            (self.project_id,)).fetchall()
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        for row in rows:
            mark = "✓" if row["analysis_status"] == "ready" else "○"
            self.video_combo.addItem(f"{mark} {Path(row['path']).name}" + (f" · {row['match_number']}" if row["match_number"] else ""),
                                     int(row["id"]))
        index = self.video_combo.findData(int(current)) if current is not None else -1
        if index < 0:
            index = next((i for i, row in enumerate(rows) if row["analysis_status"] == "ready"), 0 if rows else -1)
        self.video_combo.setCurrentIndex(index)
        self.video_combo.blockSignals(False)
        self.show_video(self.video_combo.currentData())

    def _empty(self, title: str, text: str, action: str, on_click) -> None:
        self.content.hide(); self.empty.show()
        self.empty.title.setText(title)
        self.empty.text.setText(text)
        self.empty.action.setText(action)
        self.empty.action.setVisible(bool(action))
        self._empty_action = on_click
        self.play_btn.setEnabled(False); self.scrubber.setEnabled(False)

    def show_video(self, video_id) -> None:
        self.timer.stop(); self.play_btn.setText("Play")
        self.warning.setText("")
        if self.db is None or video_id is None:
            self.video = None
            self._empty("No matches yet", "Import a match first.", "", lambda: None)
            return
        row = self.db.connection.execute("SELECT * FROM videos WHERE id=?", (int(video_id),)).fetchone()
        if row is None:
            return
        self.video = dict(row)
        calibration = self.db.calibration_for(int(video_id))
        if not calibration or len(calibration[0]) != 4:
            self._empty("No calibration — calibrate first", "Trails are placed on the field with the calibration.",
                        "Calibrate field…", lambda: self.open_calibrate.emit(int(video_id)))
            return
        if self.video["analysis_status"] != "ready":
            self._empty("Processing not finished", "Trails appear once this match has been processed.",
                        "Open the processing queue", self.open_queue.emit)
            return
        corners, zones = calibration
        notes = []
        if side_view(corners):
            notes.append("⚑ Side view — trails are less accurate here.")
        if self.video.get("tracking_mode") == "seeded_optical_flow":
            notes.append("⚑ Seeded preview trails — not accurate enough for scouting.")
        self.warning.setText(" ".join(notes))
        self.map.all_zones = dict(zones)
        self.map.trails = load_trails(self.db, int(video_id))
        frame = read_frame(Path(self.video["path"]), 0) if Path(self.video["path"]).is_file() else None
        warped = warped_field(frame, corners) if frame is not None else None
        self._picture = frame_pixmap(warped) if warped is not None else None
        last_sample = max((trail.samples[-1][0] for trail in self.map.trails if trail.samples), default=0.0)
        self.duration = max(float(self.video["duration_s"] or 0), last_sample)
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, int(self.duration * 10))
        self.scrubber.setValue(int(self.duration * 10))
        self.scrubber.blockSignals(False)
        self.map.time_s = None     # The whole match until the scrubber moves.
        self.time_label.setText(f"{format_clock(self.duration)} / {format_clock(self.duration)}")
        self._fill_legend()
        self._layers_changed()
        self.empty.hide(); self.content.show()
        self.play_btn.setEnabled(True); self.scrubber.setEnabled(True)

    def _fill_legend(self) -> None:
        while self.legend.count():
            item = self.legend.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Detach now: deleteLater alone leaves the old widget painted until the event loop runs.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        for trail in self.map.trails:
            distance = trail.distance_ft(self.map.time_s)
            box = QCheckBox(f"{trail.team} · {trail.alliance.title()} · {distance:.0f} ft · {trail.gaps()} gap(s)"
                            if trail.samples else f"{trail.team} · {trail.alliance.title()} · no positions")
            box.setChecked(trail.visible)
            box.setStyleSheet(f"QCheckBox {{ color: {trail.colour}; }}")
            box.setAccessibleName(f"Show team {trail.team}")
            box.toggled.connect(lambda on, t=trail: (setattr(t, "visible", on), self.map.update()))
            self.legend.addWidget(box)

    # ----- controls -----

    def _layers_changed(self) -> None:
        self.map.show_trails = self.layer_boxes["trails"].isChecked()
        self.map.show_heatmap = self.layer_boxes["heatmap"].isChecked()
        self.map.show_zones = self.layer_boxes["zones"].isChecked()
        self.map.background = getattr(self, "_picture", None) if self.layer_boxes["picture"].isChecked() else None
        self.map.update()

    def _scrubbed(self, value: int) -> None:
        self.map.time_s = value / 10
        self.time_label.setText(f"{format_clock(self.map.time_s)} / {format_clock(self.duration)}")
        self.map.update()
        if not self.timer.isActive():
            self._fill_legend()

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop(); self.play_btn.setText("Play"); self._fill_legend()
            return
        if self.scrubber.value() >= self.scrubber.maximum():
            self.scrubber.setValue(0)
        self.timer.start(); self.play_btn.setText("Pause")

    def _tick(self) -> None:
        value = self.scrubber.value() + int(PLAY_STEP_S * 10)
        if value >= self.scrubber.maximum():
            value = self.scrubber.maximum()
            self.timer.stop(); self.play_btn.setText("Play")
        self.scrubber.setValue(value)

    def _point_clicked(self, robot_id: int, t_s: float) -> None:
        self.scrubber.setValue(int(round(t_s * 10)))
        if self.video is not None and Path(self.video["path"]).is_file():
            frame = read_frame(Path(self.video["path"]), t_s)
            if frame is not None:
                self.preview.set_frame(frame)
        team = next((trail.team for trail in self.map.trails if trail.robot_id == robot_id), "?")
        self.time_label.setText(f"{format_clock(t_s)} / {format_clock(self.duration)} · team {team}")
