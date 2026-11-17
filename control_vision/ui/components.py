"""Thin Control Vision wrappers around qfluentwidgets components.

Each wrapper only applies Vision defaults (theme, padding, labels); the widget
itself is the library's. If a wrapper grows past ~50 lines it is hand-rolling:
use the library component directly instead.
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (BodyLabel, CaptionLabel, CardWidget, PrimaryPushButton, PushButton, StrongBodyLabel,
                            SubtitleLabel, TitleLabel, setCustomStyleSheet)

from control_vision.ui import theme

MISSING = "—"
PADDING = 16


class VisionPrimaryButton(PrimaryPushButton):
    """Volt fill, black text: the one main action of a page or card."""


class VisionSecondaryButton(PushButton):
    """Outlined grey: every other action."""


class VisionDestructiveButton(PrimaryPushButton):
    """Red fill for delete / discard; always with words that say what is lost."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        style = (f"PrimaryPushButton{{background-color:{theme.DESTRUCTIVE};border-color:{theme.DESTRUCTIVE};"
                 "color:#FFFFFF;}")
        setCustomStyleSheet(self, style, style)


class VisionCard(CardWidget):
    """A library card with the standard 16 px padding; add content to ``body``."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        self.body.setSpacing(8)


def page_header(title: str, subtitle: str = "") -> QWidget:
    """Bold 24 px title with a grey subtitle line."""
    header = QWidget()
    layout = QVBoxLayout(header)
    layout.setContentsMargins(0, 0, 0, 8)
    layout.addWidget(TitleLabel(title))
    if subtitle:
        caption = CaptionLabel(subtitle)
        caption.setTextColor(theme.TEXT_MUTED, theme.TEXT_MUTED)
        layout.addWidget(caption)
    return header


def empty_state_card(message: str, action: str, on_click: Callable[[], None]) -> VisionCard:
    """A small card at the top of a real layout with one clear action (never a full-page empty box)."""
    card = VisionCard()
    card.body.addWidget(SubtitleLabel(message))
    button = VisionPrimaryButton(action)
    button.clicked.connect(on_click)
    row = QHBoxLayout()
    row.addWidget(button)
    row.addStretch(1)
    card.body.addLayout(row)
    return card


def _score(value: int | None) -> str:
    return MISSING if value is None else str(value)


class VisionScoreStrip(VisionCard):
    """Red and Blue: our score next to the official one, never merged. Missing values show as —."""

    def __init__(self, ours: tuple[int | None, int | None], official: tuple[int | None, int | None]):
        super().__init__()
        row = QHBoxLayout()
        for index, alliance in enumerate(("red", "blue")):
            label = StrongBodyLabel(f"{alliance.title()}  {_score(ours[index])}  ·  official {_score(official[index])}")
            label.setTextColor(theme.ALLIANCE[alliance], theme.ALLIANCE[alliance])
            label.setAccessibleName(f"{alliance} score {_score(ours[index])}, official {_score(official[index])}")
            row.addWidget(label, 1)
        self.body.addLayout(row)


class VisionRobotCard(VisionCard):
    """One robot: team number in its alliance colour, and a status line."""

    def __init__(self, team: int | None, alliance: str | None, status: str = ""):
        super().__init__()
        title = StrongBodyLabel(f"Team {team}" if team is not None else f"Team {MISSING}")
        if alliance in theme.ALLIANCE:
            title.setTextColor(theme.ALLIANCE[alliance], theme.ALLIANCE[alliance])
        self.body.addWidget(title)
        self.body.addWidget(BodyLabel(f"{alliance.title()} alliance" if alliance else f"Alliance {MISSING}"))
        if status:
            self.body.addWidget(CaptionLabel(status))
