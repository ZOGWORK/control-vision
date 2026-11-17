"""The one place UI colours come from: dark graphite, Control volt accent.

Values are the Control tokens in ``control_vision.app.theme`` (DARK), so the
Fluent shell and the pages not yet moved onto it look the same. Volt replaces
Fluent's default blue accent everywhere; red and blue are reserved for the
alliances, green for confirmed, amber for needs review.
"""
from __future__ import annotations

from PySide6.QtGui import QColor
from qfluentwidgets import Theme, setTheme, setThemeColor

from control_vision.app.theme import DARK

TOKENS = DARK
VOLT = TOKENS.accent            # primary fill, active nav item, progress
ON_VOLT = TOKENS.on_accent      # text on the volt fill
SURFACE = TOKENS.surface        # cards
WINDOW = TOKENS.window          # page background
BORDER = TOKENS.border
TEXT = TOKENS.text
TEXT_MUTED = TOKENS.text_muted
RED = TOKENS.red                # red alliance only
BLUE = TOKENS.blue              # blue alliance only
CONFIRMED = TOKENS.success_fg   # green
NEEDS_REVIEW = TOKENS.pending_fg  # amber
DESTRUCTIVE = TOKENS.danger_fill
ALLIANCE = {"red": RED, "blue": BLUE}


def apply() -> None:
    """Dark Fluent theme with the volt accent. Call once, after QApplication exists. Nothing is written to disk:
    qfluentwidgets would otherwise save config/config.json in the working directory."""
    setTheme(Theme.DARK, save=False)
    setThemeColor(QColor(VOLT), save=False)
