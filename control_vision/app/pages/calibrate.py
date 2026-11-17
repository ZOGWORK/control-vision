"""Page 6 — Calibrate field: map video pixels to the 12 x 12 ft field, and draw the zones."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from control_vision.app import theme, widgets
from control_vision.app.frameview import FrameView, frame_pixmap, tokens
from control_vision.app.media import format_clock, read_frame
from control_vision.pipeline.homography import FIELD_FEET, matrix_from_corners, reprojection_error_px

CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")
ZONE_TYPES = (("Scoring — Hive, Flower, Garden", "scoring"), ("Pickup — Loading Zone", "pickup"), ("Park", "park"))
ZONE_NAMES = ("Hive", "Flower 1", "Flower 2", "Flower 3", "Flower 4", "Red Garden", "Blue Garden",
              "Red Loading Zone", "Blue Loading Zone", "Park")
# A far edge this much shorter than the near edge means the camera looks from the side.
SIDE_VIEW_RATIO = 0.6


def zone_type_for(name: str) -> str:
    lowered = name.casefold()
    if "loading" in lowered:
        return "pickup"
    if "park" in lowered:
        return "park"
    return "scoring"


def side_view(corners: list[list[float]]) -> bool:
    if len(corners) != 4:
        return False
    top = float(np.hypot(*np.subtract(corners[1], corners[0])))
    bottom = float(np.hypot(*np.subtract(corners[2], corners[3])))
    return min(top, bottom) / max(top, bottom, 1e-6) < SIDE_VIEW_RATIO


def calibration_status(corners: list[list[float]]) -> tuple[str, str, bool]:
    """(message, tone, savable) for the corners clicked so far."""
    if len(corners) < 4:
        name = CORNER_NAMES[len(corners)]
        return f"Click corner {len(corners) + 1} ({name}).", "info", False
    try:
        matrix_from_corners(corners)
        fit = reprojection_error_px(corners)
    except ValueError as exc:
        return f"✕ {exc} Re-click corners: this is not accurate enough.", "danger", False
    message = (f"✓ The corners outline the field. Corner fit {fit:.2f} px — four corners always fit exactly, so "
               "this is not an accuracy measure: check that the grid on the map follows the field's edges.")
    if side_view(corners):
        return message + " ⚑ Side view — trails will be less accurate.", "pending", True
    return message, "success", True


def scaled_calibration(corners, zones, source_size, target_size):
    """Corners and image ROIs scaled to another video's resolution; field zones are in feet and stay."""
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    scaled = [[round(x * sx, 2), round(y * sy, 2)] for x, y in corners]
    copied = {}
    for name, zone in zones.items():
        zone = dict(zone)
        if name.endswith("_roi"):
            zone["points"] = [[round(x * sx, 2), round(y * sy, 2)] for x, y in zone.get("points", [])]
        copied[name] = zone
    return scaled, copied


class CornerCanvas(FrameView):
    """Click the four corners in order; drag a numbered point to fix it."""

    changed = Signal()
    GRAB_PX = 12

    def __init__(self):
        super().__init__("Choose a video to calibrate")
        self.corners: list[list[float]] = []
        self.loupe_enabled = True
        self._dragging: int | None = None
        self.pressed.connect(self._press)
        self.moved.connect(self._move)
        self.released.connect(lambda *_: setattr(self, "_dragging", None))

    def _press(self, x: float, y: float) -> None:
        for index, (cx, cy) in enumerate(self.corners):
            point = self.to_widget(cx, cy)
            if (point - self.to_widget(x, y)).manhattanLength() <= self.GRAB_PX:
                self._dragging = index
                return
        if len(self.corners) < 4:
            self.corners.append([round(x, 2), round(y, 2)])
            self.changed.emit()
            self.update()

    def _move(self, x: float, y: float) -> None:
        if self._dragging is not None:
            self.corners[self._dragging] = [round(x, 2), round(y, 2)]
            self.changed.emit()
            self.update()

    def paint_overlay(self, painter: QPainter) -> None:
        colours = tokens()
        points = [self.to_widget(x, y) for x, y in self.corners]
        painter.setPen(QPen(QColor(colours.accent), 2))
        for start, end in zip(points, points[1:] + (points[:1] if len(points) == 4 else [])):
            painter.drawLine(start, end)
        font = QFont(painter.font()); font.setBold(True); painter.setFont(font)
        for index, point in enumerate(points):
            painter.setBrush(QColor(colours.accent))
            painter.setPen(QPen(QColor(colours.on_accent), 1))
            painter.drawEllipse(point, 10, 10)
            painter.drawText(QRectF(point.x() - 10, point.y() - 10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(index + 1))


class FieldMap(QWidget):
    """The field from above, 12 x 12 ft with 1 ft lines; drag to draw a zone."""

    rectangle_drawn = Signal(list)     # four [x, y] points in feet
    zone_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.background = None           # the frame warped to the field, when corners are valid
        self.zones: dict = {}
        self.selected: str | None = None
        self._drag_start: QPointF | None = None
        self._drag_now: QPointF | None = None
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Field map")
        self.setCursor(Qt.CursorShape.CrossCursor)

    def field_rect(self) -> QRectF:
        side = min(self.width(), self.height()) - 24
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def to_widget(self, x_ft: float, y_ft: float) -> QPointF:
        rect = self.field_rect()
        return QPointF(rect.left() + x_ft / FIELD_FEET * rect.width(), rect.top() + y_ft / FIELD_FEET * rect.height())

    def to_feet(self, point: QPointF) -> tuple[float, float]:
        rect = self.field_rect()
        x = (point.x() - rect.left()) / rect.width() * FIELD_FEET
        y = (point.y() - rect.top()) / rect.height() * FIELD_FEET
        return min(FIELD_FEET, max(0.0, x)), min(FIELD_FEET, max(0.0, y))

    def paintEvent(self, event):
        colours = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.field_rect()
        painter.fillRect(rect, QColor(colours.canvas))
        if self.background is not None:
            painter.drawPixmap(rect, self.background, QRectF(self.background.rect()))
        painter.setPen(QPen(QColor(colours.border), 1))
        for foot in range(1, int(FIELD_FEET)):
            a, b = self.to_widget(foot, 0), self.to_widget(foot, FIELD_FEET)
            painter.drawLine(a, b)
            a, b = self.to_widget(0, foot), self.to_widget(FIELD_FEET, foot)
            painter.drawLine(a, b)
        painter.setPen(QPen(QColor(colours.text), 2))
        painter.drawRect(rect)
        type_colours = {"scoring": colours.accent, "pickup": colours.success_fg, "park": colours.text_muted}
        for name, zone in self.zones.items():
            if name.endswith("_roi") or len(zone.get("points", [])) < 3:
                continue
            polygon = QPolygonF([self.to_widget(x, y) for x, y in zone["points"]])
            colour = QColor(type_colours.get(zone.get("type"), colours.text))
            fill = QColor(colour); fill.setAlpha(60)
            painter.setBrush(fill)
            painter.setPen(QPen(colour, 3 if name == self.selected else 1.5))
            painter.drawPolygon(polygon)
            painter.setPen(QColor(colours.text))
            painter.drawText(polygon.boundingRect(), Qt.AlignmentFlag.AlignCenter, name)
        if self._drag_start is not None and self._drag_now is not None:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(colours.accent), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self._drag_start, self._drag_now).normalized())
        painter.end()

    def zone_at(self, point: QPointF) -> str | None:
        for name, zone in reversed(list(self.zones.items())):
            if name.endswith("_roi") or len(zone.get("points", [])) < 3:
                continue
            if QPolygonF([self.to_widget(x, y) for x, y in zone["points"]]).containsPoint(point, Qt.FillRule.OddEvenFill):
                return name
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.field_rect().contains(event.position()):
            return
        self._drag_start = self._drag_now = event.position()

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._drag_now = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drag_start is None:
            return
        start, end = self._drag_start, event.position()
        self._drag_start = self._drag_now = None
        self.update()
        if (end - start).manhattanLength() < 8:
            name = self.zone_at(end)
            self.selected = name
            self.zone_selected.emit(name or "")
            self.update()
            return
        (x0, y0), (x1, y1) = self.to_feet(start), self.to_feet(end)
        left, right, top, bottom = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        if right - left >= 0.25 and bottom - top >= 0.25:
            self.rectangle_drawn.emit([[round(left, 3), round(top, 3)], [round(right, 3), round(top, 3)],
                                       [round(right, 3), round(bottom, 3)], [round(left, 3), round(bottom, 3)]])


def warped_field(frame: np.ndarray, corners: list[list[float]], size: int = 480):
    """The frame seen from above, size x size pixels for the 12 ft field; None if the corners are invalid."""
    try:
        matrix = matrix_from_corners(corners)
    except ValueError:
        return None
    scale = np.diag([size / FIELD_FEET, size / FIELD_FEET, 1.0])
    return cv2.warpPerspective(frame, scale @ matrix, (size, size))


class ZoneDialog(QDialog):
    def __init__(self, taken: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Name this zone")
        self.taken = taken
        form = QFormLayout(self)
        self.name = QComboBox(); self.name.setEditable(True)
        self.name.addItems([name for name in ZONE_NAMES if name not in taken])
        self.name.setAccessibleName("Zone name")
        self.kind = QComboBox()
        for text, value in ZONE_TYPES:
            self.kind.addItem(text, value)
        self.kind.setAccessibleName("Zone type")
        self.name.currentTextChanged.connect(lambda text: self.kind.setCurrentIndex(self.kind.findData(zone_type_for(text))))
        self.kind.setCurrentIndex(self.kind.findData(zone_type_for(self.name.currentText())))
        name_label = QLabel("&Name"); name_label.setBuddy(self.name)
        kind_label = QLabel("&Type"); kind_label.setBuddy(self.kind)
        form.addRow(name_label, self.name)
        form.addRow(kind_label, self.kind)
        self.problem = widgets.label("", tone="danger")
        form.addRow(self.problem)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def value(self) -> tuple[str, str]:
        return self.name.currentText().strip(), self.kind.currentData()

    def problem_with(self, name: str) -> str | None:
        if not name:
            return "Enter a name."
        if name in self.taken:
            return "That name is already used."
        if name.endswith("_roi"):
            return "Field zone names cannot end in \"_roi\"."
        return None

    def _accept(self):
        problem = self.problem_with(self.value()[0])
        if problem:
            self.problem.setText(f"✕ {problem}")
            return
        self.accept()


class CalibratePage(QWidget):
    saved = Signal(int)
    continue_identity = Signal(int)
    new_project = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        self.db = None
        self.project_id = None
        self.video: dict | None = None
        self.zones: dict = {}
        self.dirty = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        self.no_project = widgets.EmptyState("Open a project to calibrate", "Calibration is saved per video in a project.",
                                             "+ New project", self.new_project.emit)
        outer.addWidget(self.no_project, 1)
        self.body = QWidget()
        body = QVBoxLayout(self.body); body.setContentsMargins(0, 0, 0, 0); body.setSpacing(10)
        outer.addWidget(self.body, 1)

        top = QHBoxLayout()
        video_label = QLabel("&Video"); top.addWidget(video_label)
        self.video_combo = QComboBox(); self.video_combo.setAccessibleName("Video to calibrate")
        self.video_combo.setMinimumWidth(320)
        video_label.setBuddy(self.video_combo)
        self.video_combo.activated.connect(self._video_chosen)
        top.addWidget(self.video_combo)
        top.addWidget(widgets.label("Use a clearer frame", role="muted"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal); self.frame_slider.setRange(0, 1000)
        self.frame_slider.setAccessibleName("Frame to calibrate on")
        self.frame_slider.sliderReleased.connect(self._frame_changed)
        # Keys, the wheel and track clicks change the value without a release.
        self.frame_slider.valueChanged.connect(self._slider_value)
        self.frame_slider.valueChanged.connect(self._frame_label)
        top.addWidget(self.frame_slider, 1)
        self.frame_time = widgets.label("00:00", role="muted")
        top.addWidget(self.frame_time)
        top.addWidget(widgets.button("Reset corners", on_click=self.reset_corners))
        body.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal); split.setChildrenCollapsible(False)
        self.canvas = CornerCanvas()
        self.canvas.setAccessibleName("Video frame: click the four field corners")
        self.canvas.changed.connect(self._corners_changed)
        split.addWidget(self.canvas)
        right = QWidget(); right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0, 0, 0, 0)
        self.map = FieldMap()
        self.map.rectangle_drawn.connect(self._zone_drawn)
        self.map.zone_selected.connect(self._zone_selected)
        right_layout.addWidget(self.map, 1)
        zones_row = QHBoxLayout()
        zones_row.addWidget(widgets.label("Drag on the map to draw a zone; click a zone to select it.", role="muted"), 1)
        self.delete_zone_btn = widgets.button("Delete zone", "danger-outline", on_click=self.delete_zone)
        zones_row.addWidget(self.delete_zone_btn)
        right_layout.addLayout(zones_row)
        split.addWidget(right)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        body.addWidget(split, 1)

        bottom = QHBoxLayout()
        self.status = widgets.label("", wrap=True)
        self.status.setAccessibleName("Calibration status")
        bottom.addWidget(self.status, 1)
        self.reuse_btn = widgets.button("Use this calibration for…", on_click=self.reuse)
        self.save_btn = widgets.button("Save calibration", "primary", on_click=self.save)
        self.next_btn = widgets.button("Continue to robots ›", on_click=lambda: self.continue_identity.emit(self.video["id"]))
        for control in (self.reuse_btn, self.save_btn, self.next_btn):
            bottom.addWidget(control)
        body.addLayout(bottom)
        self.set_project(None, None)

    # ----- project and video -----

    def set_project(self, db, project_id) -> None:
        same = db is self.db and project_id == self.project_id
        self.db, self.project_id = db, project_id
        has = db is not None and project_id is not None
        self.no_project.setVisible(not has)
        self.body.setVisible(has)
        if not same:
            # A different project: nothing from the previous one is kept.
            self.video, self.dirty = None, False
        self.refresh_videos()

    def videos(self) -> list:
        if self.db is None or self.project_id is None:
            return []
        return self.db.connection.execute(
            "SELECT v.id,v.path,v.width,v.height,v.duration_s,v.match_number,"
            "(SELECT COUNT(*) FROM calibrations c WHERE c.video_id=v.id) AS calibrated "
            "FROM videos v WHERE v.project_id=? ORDER BY v.id", (self.project_id,)).fetchall()

    def refresh_videos(self) -> None:
        current = self.video["id"] if self.video else None
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        for row in self.videos():
            mark = "✓" if row["calibrated"] else "⚑"
            self.video_combo.addItem(f"{mark} {Path(row['path']).name}" + (f" · {row['match_number']}" if row["match_number"] else ""),
                                     int(row["id"]))
        self.video_combo.blockSignals(False)
        if current is not None and self.video_combo.findData(current) >= 0:
            self.video_combo.setCurrentIndex(self.video_combo.findData(current))
            if not self.dirty:
                # It may have been calibrated elsewhere (Set up and process) since it was loaded.
                self._load_saved(int(current))
                self._corners_changed(mark_dirty=False)
        elif self.video_combo.count():
            self.load_video(self.video_combo.itemData(0))
        else:
            self.video, self.dirty = None, False
            self.canvas.placeholder = "Import a video first"
            self.canvas.corners = []
            self.canvas.set_frame(None)
            self.zones = {}
            self.map.zones, self.map.selected, self.map.background = self.zones, None, None
            self.map.update()
        self._sync()

    def _video_chosen(self, index: int) -> None:
        video_id = self.video_combo.itemData(index)
        if self.video and video_id == self.video["id"]:
            return
        if not self.confirm_discard():
            self.video_combo.setCurrentIndex(self.video_combo.findData(self.video["id"]))
            return
        self.load_video(video_id)

    def confirm_discard(self) -> bool:
        """True when it is fine to leave this video: nothing unsaved, or the user chose."""
        if not self.dirty or self.video is None:
            return True
        answer = QMessageBox.question(
            self, "Unsaved calibration", f"Save the calibration of {Path(self.video['path']).name} first?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def load_video(self, video_id: int) -> bool:
        row = next((row for row in self.videos() if int(row["id"]) == int(video_id)), None)
        if row is None:
            return False
        self.video = dict(row)
        index = self.video_combo.findData(int(video_id))
        if index >= 0:
            self.video_combo.setCurrentIndex(index)
        self._load_saved(int(video_id))
        self.frame_slider.blockSignals(True); self.frame_slider.setValue(0); self.frame_slider.blockSignals(False)
        self._frame_label(0)
        path = Path(row["path"])
        frame = read_frame(path, 0) if path.is_file() else None
        self.canvas.placeholder = f"{path.name} cannot be read" if frame is None else ""
        self.canvas.set_frame(frame)
        self.dirty = False
        self._corners_changed(mark_dirty=False)
        return True

    def _load_saved(self, video_id: int) -> None:
        saved = self.db.calibration_for(video_id)
        corners, zones = saved if saved else ([], {})
        self.canvas.corners = [list(map(float, point)) for point in corners]
        self.zones = dict(zones)
        self.map.zones, self.map.selected = self.zones, None
        self.canvas.update()

    def _frame_label(self, value: int) -> None:
        duration = float(self.video["duration_s"] or 0) if self.video else 0.0
        self.frame_time.setText(format_clock(duration * value / 1000))

    def _slider_value(self, _value: int) -> None:
        if not self.frame_slider.isSliderDown():
            self._frame_changed()

    def _frame_changed(self) -> None:
        if self.video is None:
            return
        seconds = float(self.video["duration_s"] or 0) * self.frame_slider.value() / 1000
        frame = read_frame(Path(self.video["path"]), seconds)
        if frame is not None:
            self.canvas.set_frame(frame)
            self._update_map()

    # ----- corners and zones -----

    def reset_corners(self) -> None:
        self.canvas.corners = []
        self.canvas.update()
        self._corners_changed()

    def _corners_changed(self, mark_dirty: bool = True) -> None:
        if mark_dirty:
            self.dirty = True
        self._update_map()
        self._sync()

    def _update_map(self) -> None:
        frame = self.canvas.frame
        warped = warped_field(frame, self.canvas.corners) if frame is not None and len(self.canvas.corners) == 4 else None
        self.map.background = frame_pixmap(warped) if warped is not None else None
        self.map.update()

    def _zone_drawn(self, points: list) -> None:
        dialog = ZoneDialog(set(self.zones), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.add_zone(*dialog.value(), points)

    def add_zone(self, name: str, kind: str, points: list) -> None:
        self.zones[name] = {"type": kind, "points": points}
        self.map.zones, self.map.selected = self.zones, name
        self.dirty = True
        self.map.update()
        self._sync()

    def _zone_selected(self, name: str) -> None:
        self.map.selected = name or None
        self._sync()

    def delete_zone(self) -> None:
        if self.map.selected and self.map.selected in self.zones:
            del self.zones[self.map.selected]
            self.map.selected = None
            self.dirty = True
            self.map.update()
            self._sync()

    def _sync(self) -> None:
        has_video = self.video is not None and self.canvas.frame is not None
        message, tone, savable = calibration_status(self.canvas.corners)
        if not has_video:
            message, tone, savable = ("Import a video, then click its four field corners."
                                      if self.video is None else self.canvas.placeholder), "info", False
        if has_video and self.dirty:
            message += " Not saved yet."
        theme.set_tone(self.status, tone)
        self.status.setText(message)
        widgets.disable_with_reason(self.save_btn, has_video and savable and self.dirty,
                                    "Nothing new to save" if savable else "Click the four field corners first")
        saved = self.video is not None and bool(self.video.get("calibrated")) and not self.dirty
        widgets.disable_with_reason(self.reuse_btn, saved, "Save this calibration first")
        widgets.disable_with_reason(self.next_btn, saved, "Save this calibration first")
        widgets.disable_with_reason(self.delete_zone_btn, bool(self.map.selected), "Click a zone on the map first")

    # ----- saving -----

    def save(self) -> bool:
        if self.video is None:
            return False
        _message, _tone, savable = calibration_status(self.canvas.corners)
        if not savable:
            return False
        self.db.save_calibration(int(self.video["id"]), [list(point) for point in self.canvas.corners], self.zones)
        self.dirty = False
        self.refresh_videos()
        self.video = dict(next(row for row in self.videos() if int(row["id"]) == int(self.video["id"])))
        self._sync()
        self.saved.emit(int(self.video["id"]))
        return True

    def reuse(self) -> None:
        if self.video is None:
            return
        others = [row for row in self.videos() if int(row["id"]) != int(self.video["id"])]
        if not others:
            QMessageBox.information(self, "No other videos", "This project has no other video to use it for.")
            return
        dialog = QDialog(self); dialog.setWindowTitle("Use this calibration for")
        layout = QVBoxLayout(dialog)
        layout.addWidget(widgets.label("Only for videos from the same camera. A video that already has a calibration "
                                       "is replaced.", role="muted", wrap=True))
        choices = QListWidget(); choices.setAccessibleName("Videos to calibrate the same way")
        for row in others:
            item = QListWidgetItem(f"{Path(row['path']).name} ({row['width']}x{row['height']})"
                                   + (" · has a calibration" if row["calibrated"] else ""))
            item.setData(Qt.ItemDataRole.UserRole, int(row["id"]))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            choices.addItem(item)
        layout.addWidget(choices)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = [choices.item(i).data(Qt.ItemDataRole.UserRole) for i in range(choices.count())
                  if choices.item(i).checkState() == Qt.CheckState.Checked]
        self.apply_to(chosen)

    def apply_to(self, video_ids: list[int]) -> int:
        """Copy this calibration to other videos, scaled to their resolution."""
        rows = {int(row["id"]): row for row in self.videos()}
        source = (self.video["width"], self.video["height"])
        for video_id in video_ids:
            row = rows[int(video_id)]
            corners, zones = scaled_calibration(self.canvas.corners, self.zones, source, (row["width"], row["height"]))
            self.db.save_calibration(int(video_id), corners, zones)
            self.saved.emit(int(video_id))
        self.refresh_videos()
        if video_ids:
            theme.set_tone(self.status, "success")
            self.status.setText(f"✓ Copied to {len(video_ids)} video(s). Check each one lines up before processing.")
        return len(video_ids)
