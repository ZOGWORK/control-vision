"""Control-family design tokens for the desktop app.

Dark graphite with a volt-yellow accent, as in the Control Vision UI report:
volt marks the primary action, the active page and the active model; amber
marks anything that needs review or has low confidence; green is confirmed or
done; red and blue are the alliances (red also fills destructive buttons).
Colour is never the only signal, so status helpers also return text and icons.

One set of tokens per theme drives both the Qt palette (so native controls
such as check boxes follow the theme) and a generated stylesheet. Dark is the
default; light stays available in Settings. Nothing here touches analysis or
scoring; it only decides how things look.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget

THEME_MODES = ("system", "light", "dark")
DEFAULT_MODE = "dark"
SETTINGS_ORG = "ZOGWORK"
SETTINGS_APP = "Control Vision"
SETTINGS_KEY = "appearance/theme"
TONES = ("info", "pending", "success", "danger")
BADGES = ("accent", "pending", "success", "danger", "muted", "red", "blue")


@dataclass(frozen=True)
class Tokens:
    name: str
    window: str          # app background, graphite
    surface: str         # panels, cards, tables, inputs: one step lighter
    surface_alt: str     # table stripes, headers, hover fill
    sidebar: str
    canvas: str          # video and frame canvases: pure black
    border: str          # visible 1px border on panels and buttons
    text: str
    text_muted: str
    accent: str          # volt fill: primary buttons, progress, active page bar
    accent_hover: str
    on_accent: str       # text drawn on the accent fill (black)
    accent_text: str     # accent used as text or a thin line on the page background
    selection: str       # selected row / nav item background
    pending_fg: str      # amber: needs review, low confidence, warning
    pending_bg: str
    pending_border: str
    success_fg: str      # green: confirmed, passed, done
    success_bg: str
    danger_fg: str       # red text for errors, always with an icon and words
    danger_bg: str
    danger_fill: str     # destructive button fill
    info_bg: str
    red: str             # red alliance only
    blue: str            # blue alliance only


DARK = Tokens(
    name="dark", window="#111214", surface="#1A1C20", surface_alt="#24272D", sidebar="#0C0D0F",
    canvas="#000000", border="#3A3E46", text="#FFFFFF", text_muted="#B6BBC4",
    accent="#DFFF1A", accent_hover="#ECFF6B", on_accent="#0B0B0B", accent_text="#DFFF1A",
    selection="#2D3311",
    pending_fg="#FFB547", pending_bg="#3A2A0C", pending_border="#A86E12",
    success_fg="#4ADE80", success_bg="#0F2F1B", danger_fg="#FF7A7A", danger_bg="#3A1416",
    danger_fill="#DC2626", info_bg="#24272D", red="#EF4444", blue="#3B82F6",
)

LIGHT = Tokens(
    name="light", window="#F3F4F6", surface="#FFFFFF", surface_alt="#E9ECF0", sidebar="#FFFFFF",
    canvas="#000000", border="#C4CAD3", text="#111214", text_muted="#4A5160",
    accent="#DFFF1A", accent_hover="#C9E80A", on_accent="#0B0B0B", accent_text="#556300",
    selection="#F1FFC0",
    pending_fg="#8F5200", pending_bg="#FFF0D1", pending_border="#E0A030",
    success_fg="#15803D", success_bg="#DCFCE7", danger_fg="#B91C1C", danger_bg="#FDE2E2",
    danger_fill="#DC2626", info_bg="#E9ECF0", red="#DC2626", blue="#2563EB",
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
    value = QSettings(SETTINGS_ORG, SETTINGS_APP).value(SETTINGS_KEY, DEFAULT_MODE)
    return value if value in THEME_MODES else DEFAULT_MODE


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
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {t.border}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {t.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QWidget {{ color: {t.text}; }}
QWidget#Page, QScrollArea#PageScroll, QScrollArea#PageScroll > QWidget > QWidget {{ background: {t.window}; }}
QToolTip {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; padding: 4px 6px; }}

/* Global shell: sidebar, top bar, section tabs */
QFrame#Sidebar {{ background: {t.sidebar}; border: none; border-right: 1px solid {t.border}; }}
QLabel#AppName {{ font-size: 17px; font-weight: 700; padding: 2px 4px; }}
QLabel#AppTagline {{ color: {t.text_muted}; padding: 0 4px; font-size: 12px; }}
QListWidget#Nav {{ background: transparent; border: none; outline: none; }}
QListWidget#Nav::item {{ padding: 8px 10px; margin: 1px 0; border-radius: 4px; border-left: 3px solid transparent; }}
QListWidget#Nav::item:hover {{ background: {t.surface_alt}; }}
QListWidget#Nav::item:selected {{ background: {t.selection}; color: {t.accent_text}; border-left: 3px solid {t.accent}; font-weight: 700; }}
QListWidget#Nav::item:disabled {{ color: {t.text_muted}; }}
QFrame#TopBar {{ background: {t.sidebar}; border: none; border-bottom: 1px solid {t.border}; }}
QLabel#TopProject {{ font-size: 15px; font-weight: 700; }}
QLabel#TopMatch {{ color: {t.text_muted}; font-size: 13px; }}
QFrame#SubNav {{ background: transparent; border: none; border-bottom: 1px solid {t.border}; }}
QPushButton[subnav="true"] {{ background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0;
    padding: 8px 12px; color: {t.text_muted}; font-weight: 600; }}
QPushButton[subnav="true"]:hover {{ color: {t.text}; border-bottom: 2px solid {t.border}; background: transparent; }}
QPushButton[subnav="true"]:checked {{ color: {t.accent_text}; border-bottom: 2px solid {t.accent}; }}
QPushButton[subnav="true"]:focus {{ border-bottom: 2px solid {t.accent_text}; }}

/* Page header */
QLabel#PageTitle {{ font-size: 24px; font-weight: 700; }}
QLabel#PageContext {{ color: {t.text_muted}; }}

/* Text roles */
QLabel[role="headline"] {{ font-size: 26px; font-weight: 700; }}
QLabel[role="title"] {{ font-size: 20px; font-weight: 700; }}
QLabel[role="section"] {{ font-size: 16px; font-weight: 700; }}
QLabel[role="label"] {{ font-size: 11px; font-weight: 700; color: {t.text_muted}; letter-spacing: 1px; }}
QLabel[role="muted"] {{ color: {t.text_muted}; }}
QLabel[role="stat"] {{ font-size: 28px; font-weight: 700; }}
QLabel[role="step-number"] {{ background: {t.surface_alt}; color: {t.text_muted}; border-radius: 11px;
    min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; font-weight: 700; qproperty-alignment: AlignCenter; }}
QLabel[role="step-number"][state="done"] {{ background: {t.success_bg}; color: {t.success_fg}; }}
QLabel[role="step-number"][state="current"] {{ background: {t.accent}; color: {t.on_accent}; }}
QLabel[role="step-number"][state="running"] {{ background: {t.accent}; color: {t.on_accent}; }}
QLabel[role="step-number"][state="failed"] {{ background: {t.danger_bg}; color: {t.danger_fg}; }}
QLabel[role="step-state"] {{ color: {t.text_muted}; }}
QLabel[role="step-state"][state="done"] {{ color: {t.success_fg}; }}
QLabel[role="step-state"][state="failed"] {{ color: {t.danger_fg}; }}

/* Status callouts. "pending" (amber) marks anything a reviewer still has to check. */
QLabel[tone="info"] {{ background: {t.info_bg}; border: 1px solid {t.border}; border-radius: 6px; padding: 8px 10px; }}
QLabel[tone="pending"] {{ background: {t.pending_bg}; color: {t.pending_fg}; border: 1px solid {t.pending_border};
    border-radius: 6px; padding: 8px 10px; font-weight: 600; }}
QLabel[tone="success"] {{ background: {t.success_bg}; color: {t.success_fg}; border-radius: 6px; padding: 8px 10px; font-weight: 600; }}
QLabel[tone="danger"] {{ background: {t.danger_bg}; color: {t.danger_fg}; border-radius: 6px; padding: 8px 10px; font-weight: 600; }}

/* Badges: words plus colour, never colour alone */
QLabel[badge] {{ border-radius: 4px; padding: 2px 8px; font-size: 12px; font-weight: 700; border: 1px solid {t.border}; }}
QLabel[badge="accent"] {{ background: {t.accent}; color: {t.on_accent}; border-color: {t.accent}; }}
QLabel[badge="pending"] {{ background: {t.pending_bg}; color: {t.pending_fg}; border-color: {t.pending_border}; }}
QLabel[badge="success"] {{ background: {t.success_bg}; color: {t.success_fg}; border-color: {t.success_fg}; }}
QLabel[badge="danger"] {{ background: {t.danger_bg}; color: {t.danger_fg}; border-color: {t.danger_fg}; }}
QLabel[badge="muted"] {{ background: {t.surface_alt}; color: {t.text_muted}; }}
QLabel[badge="red"] {{ background: {t.red}; color: #FFFFFF; border-color: {t.red}; }}
QLabel[badge="blue"] {{ background: {t.blue}; color: #FFFFFF; border-color: {t.blue}; }}

/* Cards, panels and group boxes */
QFrame[card="true"] {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 8px; }}
QFrame[card="true"] QLabel {{ background: transparent; }}
QFrame[card="true"][interactive="true"]:hover {{ border: 1px solid {t.text_muted}; background: {t.surface_alt}; }}
QFrame[card="true"][selected="true"] {{ border: 2px solid {t.accent}; }}
QFrame[card="true"][alliance="red"] {{ border: 2px solid {t.red}; }}
QFrame[card="true"][alliance="blue"] {{ border: 2px solid {t.blue}; }}
QLabel#Thumbnail {{ background: {t.canvas}; color: #FFFFFF; border-radius: 6px; font-weight: 700; }}
QGroupBox {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 8px;
    margin-top: 22px; padding: 12px 12px 10px 12px; font-weight: 700; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 4px; top: 2px; padding: 0 4px; }}
QGroupBox QLabel, QGroupBox QCheckBox {{ font-weight: normal; }}

/* Buttons: every button has a border or a fill, and a visible hover. */
QPushButton, QToolButton {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; border-radius: 4px;
    padding: 6px 14px; min-height: 18px; }}
QPushButton:hover, QToolButton:hover {{ background: {t.surface_alt}; border-color: {t.text_muted}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t.selection}; }}
QPushButton:focus, QToolButton:focus {{ border: 2px solid {t.accent_text}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t.text_muted}; background: {t.surface_alt}; border: 1px dashed {t.border}; }}
QPushButton[variant="primary"] {{ background: {t.accent}; color: {t.on_accent}; border: 1px solid {t.accent}; font-weight: 700; }}
QPushButton[variant="primary"]:hover {{ background: {t.accent_hover}; border-color: {t.accent_hover}; }}
QPushButton[variant="primary"]:focus {{ border: 2px solid {t.text}; }}
QPushButton[variant="primary"]:disabled {{ background: {t.surface_alt}; color: {t.text_muted}; border: 1px dashed {t.border}; font-weight: normal; }}
QPushButton[variant="danger"] {{ background: {t.danger_fill}; color: #FFFFFF; border: 1px solid {t.danger_fill}; font-weight: 700; }}
QPushButton[variant="danger"]:hover {{ background: {t.danger_fg}; border-color: {t.danger_fg}; color: #000000; }}
QPushButton[variant="danger-outline"] {{ color: {t.danger_fg}; border: 1px solid {t.danger_fg}; background: transparent; }}
QPushButton[variant="danger-outline"]:hover {{ background: {t.danger_bg}; }}
QPushButton[variant="success"] {{ background: {t.success_fg}; color: #000000; border: 1px solid {t.success_fg}; font-weight: 700; }}
QPushButton[variant="success"]:hover {{ background: {t.success_bg}; color: {t.success_fg}; }}
QPushButton[variant="link"] {{ background: transparent; border: 1px solid transparent; color: {t.accent_text}; text-decoration: underline; padding: 2px 4px; }}
QPushButton[variant="link"]:hover {{ border: 1px solid {t.border}; }}
QPushButton[alliance="red"]:checked {{ background: {t.red}; color: #FFFFFF; border-color: {t.red}; font-weight: 700; }}
QPushButton[alliance="blue"]:checked {{ background: {t.blue}; color: #FFFFFF; border-color: {t.blue}; font-weight: 700; }}

/* Inputs. Combo and spin boxes keep the Fusion frame and arrows, drawn from the palette. */
QLineEdit, QPlainTextEdit, QTextEdit, QDateEdit {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 4px;
    padding: 5px 8px; min-height: 20px; selection-background-color: {t.accent}; selection-color: {t.on_accent}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QDateEdit:focus {{ border: 2px solid {t.accent_text}; }}
QLineEdit:disabled {{ color: {t.text_muted}; background: {t.surface_alt}; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ min-height: 24px; }}
QComboBox QAbstractItemView {{ background: {t.surface}; border: 1px solid {t.border}; selection-background-color: {t.selection};
    selection-color: {t.text}; outline: none; }}

/* Tables and lists */
QTableWidget, QTableView, QListWidget, QTextBrowser {{ background: {t.surface}; alternate-background-color: {t.surface_alt};
    border: 1px solid {t.border}; border-radius: 8px; gridline-color: {t.border};
    selection-background-color: {t.selection}; selection-color: {t.text}; outline: none; }}
QTableWidget::item, QTableView::item, QListWidget::item {{ padding: 4px 6px; }}
QListWidget::item:selected {{ background: {t.selection}; color: {t.text}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: {t.surface_alt}; color: {t.text_muted}; border: none; border-bottom: 1px solid {t.border};
    padding: 7px 8px; font-weight: 700; }}
QTableCornerButton::section {{ background: {t.surface_alt}; border: none; }}

/* Progress and seek bar */
QProgressBar {{ background: {t.surface_alt}; border: 1px solid {t.border}; border-radius: 4px; max-height: 10px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 3px; }}
QSlider::groove:horizontal {{ height: 6px; background: {t.surface_alt}; border: 1px solid {t.border}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {t.surface}; border: 2px solid {t.accent}; width: 14px; height: 14px; margin: -6px 0; border-radius: 8px; }}

QStatusBar {{ background: {t.sidebar}; border-top: 1px solid {t.border}; color: {t.text_muted}; }}
QStatusBar QLabel {{ color: {t.text_muted}; padding: 0 6px; }}
QStatusBar QLabel[tone="danger"] {{ padding: 1px 8px; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: transparent; }}
"""


def apply_theme(app: QApplication, mode: str = "system") -> Tokens:
    """Style the whole application for ``mode`` and return the tokens used."""
    tokens = resolve_tokens(mode, app)
    app.setStyle("Fusion")
    font = QFont("Segoe UI")
    font.setPointSizeF(10)
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


def set_badge(label: QWidget, kind: str, text: str) -> None:
    """Show a status badge: words plus a colour, never colour alone."""
    if kind not in BADGES:
        raise ValueError(f"Unknown badge: {kind}")
    label.setText(text)
    if label.property("badge") != kind:
        label.setProperty("badge", kind)
        label.style().unpolish(label)
        label.style().polish(label)


def set_state(widget: QWidget, state: str | None) -> None:
    if widget.property("state") == state:
        return
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


STATUS_LABELS = {"flagged": "⚑ Needs review", "confirmed": "✓ Confirmed", "rejected": "✕ Rejected"}


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
