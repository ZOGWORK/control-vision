"""A video frame you can click on, in video pixels, with a magnifier for precise clicks."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget

from control_vision.app import theme


def tokens():
    return theme.resolve_tokens(QApplication.instance().property("control_vision_theme") or theme.DEFAULT_MODE)


def frame_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy())


class FrameView(QWidget):
    """Shows one frame letterboxed; mouse events are reported in video pixels."""

    pressed = Signal(float, float)
    moved = Signal(float, float)
    released = Signal(float, float)

    LOUPE = 140      # magnifier size on screen
    ZOOM = 3

    def __init__(self, placeholder: str = ""):
        super().__init__()
        self.frame: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self.placeholder = placeholder
        self.loupe_enabled = False
        self._hover: QPointF | None = None
        self.setMinimumSize(400, 225)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_frame(self, frame: np.ndarray | None) -> None:
        self.frame = frame
        self._pixmap = frame_pixmap(frame) if frame is not None else None
        self.update()

    def frame_rect(self) -> QRectF:
        """Where the frame is drawn inside the widget, keeping its aspect ratio."""
        if self.frame is None:
            return QRectF()
        height, width = self.frame.shape[:2]
        scale = min(self.width() / width, self.height() / height)
        drawn_w, drawn_h = width * scale, height * scale
        return QRectF((self.width() - drawn_w) / 2, (self.height() - drawn_h) / 2, drawn_w, drawn_h)

    def to_widget(self, x: float, y: float) -> QPointF:
        rect = self.frame_rect()
        height, width = self.frame.shape[:2]
        return QPointF(rect.left() + x * rect.width() / width, rect.top() + y * rect.height() / height)

    def to_video(self, point: QPointF) -> tuple[float, float] | None:
        if self.frame is None:
            return None
        rect = self.frame_rect()
        if not rect.contains(point):
            return None
        height, width = self.frame.shape[:2]
        return ((point.x() - rect.left()) * width / rect.width(), (point.y() - rect.top()) * height / rect.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colours = tokens()
        painter.fillRect(self.rect(), QColor(colours.canvas))
        if self._pixmap is None:
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.placeholder)
            return
        painter.drawPixmap(self.frame_rect(), self._pixmap, QRectF(self._pixmap.rect()))
        self.paint_overlay(painter)
        if self.loupe_enabled and self._hover is not None:
            self._paint_loupe(painter)

    def paint_overlay(self, painter: QPainter) -> None:
        """Subclasses draw points and boxes here."""

    def _paint_loupe(self, painter: QPainter) -> None:
        video = self.to_video(self._hover)
        if video is None:
            return
        source_size = self.LOUPE / self.ZOOM / (self.frame_rect().width() / self.frame.shape[1])
        source = QRectF(video[0] - source_size / 2, video[1] - source_size / 2, source_size, source_size)
        target = QRectF(self.width() - self.LOUPE - 8, 8, self.LOUPE, self.LOUPE)
        painter.fillRect(target, QColor(tokens().canvas))
        painter.drawPixmap(target, self._pixmap, source)
        painter.setPen(QPen(QColor(tokens().accent), 1))
        painter.drawRect(target)
        centre = target.center()
        painter.drawLine(QPointF(centre.x() - 10, centre.y()), QPointF(centre.x() + 10, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y() - 10), QPointF(centre.x(), centre.y() + 10))

    def mousePressEvent(self, event):
        video = self.to_video(event.position())
        if event.button() == Qt.MouseButton.LeftButton and video is not None:
            self.pressed.emit(*video)

    def mouseMoveEvent(self, event):
        self._hover = event.position()
        video = self.to_video(event.position())
        if video is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.moved.emit(*video)
        self.update()

    def clamped_video(self, point: QPointF) -> tuple[float, float] | None:
        """Video pixels for a point, moved onto the frame's edge when it is outside."""
        if self.frame is None:
            return None
        rect = self.frame_rect()
        inside = QPointF(min(max(point.x(), rect.left()), rect.right()), min(max(point.y(), rect.top()), rect.bottom()))
        height, width = self.frame.shape[:2]
        return ((inside.x() - rect.left()) * width / rect.width(), (inside.y() - rect.top()) * height / rect.height())

    def mouseReleaseEvent(self, event):
        # Always report a release, even over the black bars, so a drag never stays active.
        video = self.clamped_video(event.position())
        if event.button() == Qt.MouseButton.LeftButton and video is not None:
            self.released.emit(*video)

    def leaveEvent(self, event):
        self._hover = None
        self.update()
