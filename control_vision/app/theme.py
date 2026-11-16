"""Light and dark themes for the desktop app.

One set of colour tokens per theme drives both the Qt palette (so native
controls such as check boxes follow the theme) and a generated stylesheet.
Nothing here touches analysis or scoring; it only decides how things look.

Accent is a honey yellow. Anything a reviewer still has to check (flagged
events, low-confidence values, pending rulings, OCR proposals) uses a separate
orange "pending" tone so it never blends into the accent.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget

THEME_MODES = ("system", "light", "dark")
SETTINGS_ORG = "ZOGWORK"
SETTINGS_APP = "Control Vision"
SETTINGS_KEY = "appearance/theme"
TONES = ("info", "pending", "success", "danger")


@dataclass(frozen=True)
class Tokens:
    name: str
    window: str          # page background
    surface: str         # cards, tables, inputs
    surface_alt: str     # table stripes, headers, hover
    sidebar: str
    border: str
    text: str
    text_muted: str
    accent: str          # yellow fill: primary buttons, progress, selection marker
    accent_hover: str
    on_accent: str       # text drawn on the accent fill
    accent_text: str     # accent used as text or a thin line on the page background
    selection: str       # selected row / nav item background
    pending_fg: str
    pending_bg: str
    pending_border: str
    success_fg: str
    success_bg: str
    danger_fg: str
    danger_bg: str
    info_bg: str


LIGHT = Tokens(
    name="light", window="#F5F6F8", surface="#FFFFFF", surface_alt="#F0F2F5", sidebar="#FFFFFF",
    border="#DCE0E6", text="#1A202C", text_muted="#5A6372",
    accent="#F5C211", accent_hover="#E3B000", on_accent="#1F1600", accent_text="#8A6300",
    selection="#FFF3C4",
    pending_fg="#B54708", pending_bg="#FFEBD6", pending_border="#F7A867",
    success_fg="#16753A", success_bg="#DDF5E5", danger_fg="#B42318", danger_bg="#FDE4E1",
    info_bg="#EEF1F5",
)

DARK = Tokens(
    name="dark", window="#101318", surface="#181C23", surface_alt="#1F242D", sidebar="#14171D",
    border="#2C323D", text="#E7EAF0", text_muted="#9AA3B2",
    accent="#F5C518", accent_hover="#FFD43B", on_accent="#1A1300", accent_text="#F5C518",
    selection="#3A3214",
    pending_fg="#FDB474", pending_bg="#3A2111", pending_border="#9A4D16",
    success_fg="#7FE0A2", success_bg="#11301D", danger_fg="#FCA19A", danger_bg="#3A1513",
    info_bg="#1F242D",
)


def system_prefers_dark(app: QApplication | None = None) -> bool:
    app = app or QApplication.instance()
    if app is None:
        return False
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:  # Qt older than 6.5 has no colour-scheme hint.
        return False


def resolve_tokens(mode: str, app: QApplication | None = None) -> Tokens:
    """Tokens for "light", "dark" or "system" (unknown modes follow the system)."""
    if mode == "light":
        return LIGHT
    if mode == "dark":
        return DARK
    return DARK if system_prefers_dark(app) else LIGHT


def saved_mode() -> str:
    value = QSettings(SETTINGS_ORG, SETTINGS_APP).value(SETTINGS_KEY, "system")
    return value if value in THEME_MODES else "system"


def save_mode(mode: str) -> None:
    if mode not in THEME_MODES:
        raise ValueError(f"Unknown theme mode: {mode}")
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(SETTINGS_KEY, mode)


def build_palette(t: Tokens) -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: t.window, QPalette.ColorRole.WindowText: t.text,
        QPalette.ColorRole.Base: t.surface, QPalette.ColorRole.AlternateBase: t.surface_alt,
        QPalette.ColorRole.Text: t.text, QPalette.ColorRole.Button: t.surface,
        QPalette.ColorRole.ButtonText: t.text, QPalette.ColorRole.Highlight: t.accent,
        QPalette.ColorRole.HighlightedText: t.on_accent, QPalette.ColorRole.ToolTipBase: t.surface,
        QPalette.ColorRole.ToolTipText: t.text, QPalette.ColorRole.PlaceholderText: t.text_muted,
        QPalette.ColorRole.Link: t.accent_text, QPalette.ColorRole.Mid: t.border,
    }
    for role, colour in roles.items():
        palette.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(t.text_muted))
    return palette


def build_stylesheet(t: Tokens) -> str:
    return f"""
QMainWindow, QDialog {{ background: {t.window}; }}
QWidget {{ color: {t.text}; }}
QWidget#Page, QScrollArea#PageScroll, QScrollArea#PageScroll > QWidget > QWidget {{ background: {t.window}; }}
QToolTip {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; padding: 4px 6px; }}

/* Sidebar navigation */
QFrame#Sidebar {{ background: {t.sidebar}; border: none; border-right: 1px solid {t.border}; }}
QLabel#AppName {{ font-size: 17px; font-weight: 600; padding: 2px 4px; }}
QLabel#AppTagline {{ color: {t.text_muted}; padding: 0 4px; }}
QListWidget#Nav {{ background: transparent; border: none; outline: none; }}
QListWidget#Nav::item {{ padding: 7px 10px; margin: 1px 0; border-radius: 6px; border-left: 3px solid transparent; }}
QListWidget#Nav::item:hover {{ background: {t.surface_alt}; }}
QListWidget#Nav::item:selected {{ background: {t.selection}; color: {t.text}; border-left: 3px solid {t.accent}; }}
QListWidget#Nav::item:disabled {{ color: {t.text_muted}; }}

/* Page header */
QLabel#PageTitle {{ font-size: 20px; font-weight: 600; }}
QLabel#PageContext {{ color: {t.text_muted}; }}

/* Text roles */
QLabel[role="title"] {{ font-size: 20px; font-weight: 600; }}
QLabel[role="section"] {{ font-size: 13px; font-weight: 600; }}
QLabel[role="muted"] {{ color: {t.text_muted}; }}
QLabel[role="step-number"] {{ background: {t.surface_alt}; color: {t.text_muted}; border-radius: 11px;
    min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; font-weight: 600; qproperty-alignment: AlignCenter; }}
QLabel[role="step-number"][state="done"] {{ background: {t.success_bg}; color: {t.success_fg}; }}
QLabel[role="step-number"][state="current"] {{ background: {t.accent}; color: {t.on_accent}; }}
QLabel[role="step-number"][state="running"] {{ background: {t.accent}; color: {t.on_accent}; }}
QLabel[role="step-number"][state="failed"] {{ background: {t.danger_bg}; color: {t.danger_fg}; }}
QLabel[role="step-state"] {{ color: {t.text_muted}; }}
QLabel[role="step-state"][state="done"] {{ color: {t.success_fg}; }}
QLabel[role="step-state"][state="failed"] {{ color: {t.danger_fg}; }}

/* Status callouts. "pending" marks anything a reviewer still has to check. */
QLabel[tone="info"] {{ background: {t.info_bg}; border: 1px solid {t.border}; border-radius: 6px; padding: 7px 10px; }}
QLabel[tone="pending"] {{ background: {t.pending_bg}; color: {t.pending_fg}; border: 1px solid {t.pending_border};
    border-radius: 6px; padding: 7px 10px; font-weight: 600; }}
QLabel[tone="success"] {{ background: {t.success_bg}; color: {t.success_fg}; border-radius: 6px; padding: 7px 10px; }}
QLabel[tone="danger"] {{ background: {t.danger_bg}; color: {t.danger_fg}; border-radius: 6px; padding: 7px 10px; }}

/* Cards and group boxes */
QFrame[card="true"] {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 10px; }}
QFrame[card="true"] QLabel {{ background: transparent; }}
QGroupBox {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 10px;
    margin-top: 22px; padding: 12px 12px 10px 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 4px; top: 2px; padding: 0 4px; }}
QGroupBox QLabel, QGroupBox QCheckBox {{ font-weight: normal; }}

/* Buttons */
QPushButton {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 6px; padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ background: {t.surface_alt}; border-color: {t.accent_hover}; }}
QPushButton:pressed {{ background: {t.selection}; }}
QPushButton:focus {{ border: 1px solid {t.accent_text}; }}
QPushButton:disabled {{ color: {t.text_muted}; background: {t.surface_alt}; border-color: {t.border}; }}
QPushButton[variant="primary"] {{ background: {t.accent}; color: {t.on_accent}; border: 1px solid {t.accent_hover}; font-weight: 600; }}
QPushButton[variant="primary"]:hover {{ background: {t.accent_hover}; }}
QPushButton[variant="primary"]:disabled {{ background: {t.surface_alt}; color: {t.text_muted}; border-color: {t.border}; font-weight: normal; }}
QPushButton[variant="danger"] {{ color: {t.danger_fg}; }}
QPushButton[variant="danger"]:hover {{ background: {t.danger_bg}; border-color: {t.danger_fg}; }}

/* Inputs. Combo and spin boxes keep Fusion's own frame and arrows, drawn from the palette. */
QLineEdit {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 6px;
    padding: 4px 8px; min-height: 20px; selection-background-color: {t.accent}; selection-color: {t.on_accent}; }}
QLineEdit:focus {{ border: 1px solid {t.accent_text}; }}
QLineEdit:disabled {{ color: {t.text_muted}; background: {t.surface_alt}; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ min-height: 24px; }}
QComboBox QAbstractItemView {{ background: {t.surface}; border: 1px solid {t.border}; selection-background-color: {t.selection};
    selection-color: {t.text}; outline: none; }}

/* Tables and lists */
QTableWidget, QListWidget, QTextBrowser {{ background: {t.surface}; alternate-background-color: {t.surface_alt};
    border: 1px solid {t.border}; border-radius: 8px; gridline-color: {t.border};
    selection-background-color: {t.selection}; selection-color: {t.text}; outline: none; }}
QTableWidget::item, QListWidget::item {{ padding: 4px 6px; }}
QListWidget::item:selected {{ background: {t.selection}; color: {t.text}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: {t.surface_alt}; color: {t.text_muted}; border: none; border-bottom: 1px solid {t.border};
    padding: 7px 8px; font-weight: 600; }}
QTableCornerButton::section {{ background: {t.surface_alt}; border: none; }}

/* Progress and seek bar */
QProgressBar {{ background: {t.surface_alt}; border: 1px solid {t.border}; border-radius: 5px; max-height: 10px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 4px; }}
QSlider::groove:horizontal {{ height: 6px; background: {t.surface_alt}; border: 1px solid {t.border}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {t.surface}; border: 2px solid {t.accent_hover}; width: 14px; height: 14px; margin: -6px 0; border-radius: 8px; }}

QStatusBar {{ background: {t.sidebar}; border-top: 1px solid {t.border}; color: {t.text_muted}; }}
QStatusBar QLabel {{ color: {t.text_muted}; padding: 0 6px; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: transparent; }}
"""


def apply_theme(app: QApplication, mode: str = "system") -> Tokens:
    """Style the whole application for ``mode`` and return the tokens used."""
    tokens = resolve_tokens(mode, app)
    app.setStyle("Fusion")
    font = QFont("Segoe UI")
    font.setPointSizeF(9.5)
    app.setFont(font)
    app.setPalette(build_palette(tokens))
    app.setStyleSheet(build_stylesheet(tokens))
    app.setProperty("control_vision_theme", tokens.name)
    return tokens


def set_tone(widget: QWidget, tone: str | None) -> None:
    """Give a label one of TONES (or none) and restyle it."""
    if tone is not None and tone not in TONES:
        raise ValueError(f"Unknown tone: {tone}")
    if widget.property("tone") == tone:
        return
    widget.setProperty("tone", tone)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_state(widget: QWidget, state: str | None) -> None:
    if widget.property("state") == state:
        return
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


STATUS_LABELS = {"flagged": "⚑ Pending review", "confirmed": "✓ Confirmed", "rejected": "✕ Rejected"}


def status_label(status: str) -> str:
    """Reviewer-facing text for an event status; unknown statuses are shown as stored."""
    return STATUS_LABELS.get(str(status), str(status))


def confidence_label(confidence: str) -> str:
    text = str(confidence)
    return "⚑ low" if text == "low" else text


def row_colours(t: Tokens, status: str, confidence: str | None = None) -> tuple[str | None, str | None]:
    """(background, foreground) for a table row; (None, None) keeps the defaults.

    Flagged events and low-confidence rows use the pending tone, which is
    distinct from the accent; rejected rows are muted.
    """
    if status == "rejected":
        return None, t.text_muted
    if status == "flagged" or confidence == "low":
        return t.pending_bg, t.pending_fg
    return None, None
