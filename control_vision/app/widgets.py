"""Small shared widgets for the Control Vision pages.

Every page is built from these so buttons, cards, badges and empty states look
and behave the same everywhere: buttons always have a border or a fill, status
always has words next to its colour, and a list with nothing in it always says
why and offers one action.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from control_vision.app import theme


def card(interactive: bool = False) -> QFrame:
    frame = QFrame()
    frame.setProperty("card", True)
    if interactive:
        frame.setProperty("interactive", True)
        frame.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        frame.setCursor(Qt.CursorShape.PointingHandCursor)
    return frame


class ClickableCard(QFrame):
    """A card that acts like a button: click it, or focus it and press Enter or Space."""

    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setProperty("card", True)
        self.setProperty("interactive", True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


def label(text: str = "", role: str | None = None, tone: str | None = None, wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if role:
        widget.setProperty("role", role)
    if tone:
        widget.setProperty("tone", tone)
    widget.setWordWrap(wrap)
    return widget


def section_label(text: str) -> QLabel:
    """Small grey uppercase label, for section labels only."""
    return label(text.upper(), role="label")


def button(text: str, variant: str | None = None, tooltip: str | None = None,
           on_click: Callable | None = None) -> QPushButton:
    widget = QPushButton(text)
    if variant:
        widget.setProperty("variant", variant)
    if tooltip:
        widget.setToolTip(tooltip)
    if on_click is not None:
        widget.clicked.connect(lambda *_: on_click())
    return widget


def badge(kind: str, text: str) -> QLabel:
    widget = QLabel()
    widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    theme.set_badge(widget, kind, text)
    widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return widget


PROJECT_STATUS_BADGES = {
    "Not processed": ("muted", "○ Not processed"),
    "Processing": ("accent", "◐ Processing"),
    "Ready to review": ("pending", "⚑ Ready to review"),
    "Reviewed": ("success", "✓ Reviewed"),
    "Exported": ("success", "✓ Exported"),
}


def status_badge(status: str) -> QLabel:
    kind, text = PROJECT_STATUS_BADGES.get(status, ("muted", status))
    return badge(kind, text)


def disable_with_reason(widget: QWidget, enabled: bool, reason: str) -> None:
    """Disable a control and say why on hover; restore the tooltip when enabled."""
    if widget.property("enabled_tooltip") is None:
        widget.setProperty("enabled_tooltip", widget.toolTip())
    widget.setEnabled(enabled)
    widget.setToolTip(widget.property("enabled_tooltip") if enabled else reason)


class StatCard(QFrame):
    def __init__(self, title: str, value: str = "0", note: str = ""):
        super().__init__()
        self.setProperty("card", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)
        layout.addWidget(section_label(title))
        self.value = label(value, role="stat")
        self.note = label(note, role="muted", wrap=True)
        self.value.setAccessibleName(title)
        layout.addWidget(self.value)
        layout.addWidget(self.note)

    def set(self, value: str, note: str = "") -> None:
        self.value.setText(value)
        self.note.setText(note)


class EmptyState(QFrame):
    """Centred message with one clear action, so no page or list is ever blank."""

    def __init__(self, title: str, text: str = "", action_text: str | None = None,
                 on_action: Callable | None = None, primary: bool = True):
        super().__init__()
        self.setProperty("card", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(10)
        layout.addStretch(1)
        self.title = label(title, role="title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text = label(text, role="muted", wrap=True)
        self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)
        layout.addWidget(self.text)
        self.action = None
        if action_text:
            self.action = button(action_text, "primary" if primary else None, on_click=on_action)
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(self.action)
            row.addStretch(1)
            layout.addLayout(row)
        layout.addStretch(1)


class Thumbnail(QLabel):
    """16:9 still: a frame from a video, or black with white text when there is none."""

    def __init__(self, width: int = 288):
        super().__init__()
        self.setObjectName("Thumbnail")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setFixedSize(QSize(width, round(width * 9 / 16)))

    def show_placeholder(self, text: str) -> None:
        self.clear()
        self.setText(text)

    def show_image(self, image: QImage) -> None:
        self.setText("")
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pixmap)


def video_still(path: Path | None, at_fraction: float = 0.1) -> QImage | None:
    """One frame from a video as a QImage, or None if it cannot be read."""
    if path is None or not Path(path).is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(count * at_fraction))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        return None
    frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()


class PlannedPage(QWidget):
    """A page from the UI report whose feature is not built yet.

    It says what the page will do, why it is not available, and offers the
    nearest thing that works today, so the page is never blank.
    """

    def __init__(self, title: str, purpose: str, status: str, action_text: str | None = None,
                 on_action: Callable | None = None):
        super().__init__()
        self.setObjectName("Page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.empty = EmptyState(title, purpose, action_text, on_action, primary=False)
        self.status = label(status, tone="pending", wrap=True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.layout().insertWidget(3, self.status)
        layout.addWidget(self.empty)


class FlowGrid(QWidget):
    """A fixed-column grid that rebuilds from a list of widgets."""

    def __init__(self, columns: int = 3):
        super().__init__()
        self.columns = columns
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(16)

    def set_widgets(self, widgets: list[QWidget]) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for index, widget in enumerate(widgets):
            self.grid.addWidget(widget, index // self.columns, index % self.columns)
        for column in range(self.columns):
            self.grid.setColumnStretch(column, 1)
