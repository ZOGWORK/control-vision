"""Page 1 — Home dashboard: what is recent, what needs attention, where to start."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from control_vision.app import widgets
from control_vision.app.library import PROJECT_STATUSES, HomeStats, ModelVersion, ProjectSummary


def greeting_line(today: datetime, stats: HomeStats) -> str:
    """For example "Sunday, Nov 15 — 3 projects, 2 ready to review"."""
    day = f"{today:%A}, {today:%b} {today.day}"
    projects = f"{stats.projects} project" + ("" if stats.projects == 1 else "s")
    return f"{day} — {projects}, {stats.ready_to_review} ready to review"


def model_stat(model: ModelVersion | None) -> tuple[str, str]:
    """Average mAP is the active model's mAP50, with the version named; never invented."""
    if model is None:
        return "—", "No active model. Add one under Models."
    if model.map50 is None:
        return "—", f"{model.name}: not measured yet"
    return f"{model.map50:.2f}", f"mAP50 · {model.name}"


class ProjectCard(QWidget):
    clicked = Signal(object)

    def __init__(self, summary: ProjectSummary, still=None):
        super().__init__()
        self.summary = summary
        frame = widgets.ClickableCard()
        frame.setAccessibleName(f"Open project {summary.display_name}")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        self.thumbnail = widgets.Thumbnail(272)
        placeholder = summary.event_name or summary.display_name
        if summary.missing:
            self.thumbnail.show_placeholder(f"{placeholder}\nFile missing")
        elif still is not None:
            self.thumbnail.show_image(still)
        else:
            self.thumbnail.show_placeholder(placeholder)
        layout.addWidget(self.thumbnail)
        title = widgets.label(summary.display_name, role="section")
        title.setWordWrap(True)
        layout.addWidget(title)
        event = " · ".join(part for part in (summary.event_name, summary.event_code, summary.event_date) if part)
        layout.addWidget(widgets.label(event or "No event details", role="muted"))
        row = QHBoxLayout()
        row.addWidget(widgets.badge("danger", "✕ Missing file") if summary.missing else widgets.status_badge(summary.status))
        row.addStretch(1)
        layout.addLayout(row)
        matches = f"{summary.matches} match" + ("" if summary.matches == 1 else "es")
        layout.addWidget(widgets.label(f"{matches} · model {summary.model_name or '—'}", role="muted"))
        frame.clicked.connect(lambda: self.clicked.emit(summary.path))
        self.frame = frame


class HomePage(QWidget):
    new_project = Signal()
    import_video = Signal()
    resume = Signal()
    open_project = Signal(object)
    open_processing = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        self._stills: dict[tuple[str, float], object] = {}
        self._summaries: list[ProjectSummary] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)

        top = QHBoxLayout()
        self.greeting = widgets.label("", role="title")
        self.greeting.setAccessibleName("Today")
        top.addWidget(self.greeting, 1)
        self.new_btn = widgets.button("+ New project", "primary", on_click=self.new_project.emit)
        self.import_btn = widgets.button("Import video", on_click=self.import_video.emit)
        self.resume_btn = widgets.button("Resume last project", on_click=self.resume.emit)
        for control in (self.new_btn, self.import_btn, self.resume_btn):
            top.addWidget(control)
        outer.addLayout(top)

        stats = QHBoxLayout()
        stats.setSpacing(16)
        self.stat_matches = widgets.StatCard("Matches analysed", "0", "Processed match videos")
        self.stat_cycles = widgets.StatCard("Cycles counted", "0", "Zone-trip cycles in analysed matches")
        self.stat_map = widgets.StatCard("Average mAP", "—", "No active model")
        self.stat_hours = widgets.StatCard("Hours processed", "0.0", "Of match video")
        for card in (self.stat_matches, self.stat_cycles, self.stat_map, self.stat_hours):
            stats.addWidget(card, 1)
        outer.addLayout(stats)

        self.activity = widgets.ClickableCard()
        self.activity.setAccessibleName("Open Processing")
        activity_layout = QHBoxLayout(self.activity)
        activity_layout.setContentsMargins(16, 10, 16, 10)
        self.activity_text = widgets.label("", wrap=True)
        activity_layout.addWidget(self.activity_text, 1)
        activity_layout.addWidget(widgets.label("Open Processing ›", role="muted"))
        self.activity.clicked.connect(self.open_processing.emit)
        self.activity.hide()
        outer.addWidget(self.activity)

        filters = QHBoxLayout()
        filters.addWidget(widgets.label("Recent projects", role="section"), 1)
        self.season_filter = QComboBox()
        self.season_filter.addItem("All seasons", None)
        self.season_filter.addItem("BIOBUZZ 2026-27", "BIOBUZZ 2026-27")
        self.season_filter.setAccessibleName("Filter recent projects by season")
        self.status_filter = QComboBox()
        self.status_filter.addItem("All statuses", None)
        for status in PROJECT_STATUSES:
            self.status_filter.addItem(status, status)
        self.status_filter.setAccessibleName("Filter recent projects by status")
        season_label = QLabel("&Season"); season_label.setBuddy(self.season_filter)
        status_label = QLabel("S&tatus"); status_label.setBuddy(self.status_filter)
        for widget in (season_label, self.season_filter, status_label, self.status_filter):
            filters.addWidget(widget)
        outer.addLayout(filters)
        self.season_filter.currentIndexChanged.connect(self._render)
        self.status_filter.currentIndexChanged.connect(self._render)

        self.grid = widgets.FlowGrid(3)
        self.grid_holder = QWidget()
        holder_layout = QVBoxLayout(self.grid_holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.addWidget(self.grid)
        holder_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setObjectName("PageScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_holder)
        self.scroll = scroll
        outer.addWidget(scroll, 1)

        self.empty = widgets.EmptyState(
            "No projects yet — Create your first project",
            "Import a match video, calibrate the field, and Vision will count cycles.",
            "+ New project", self.new_project.emit)
        outer.addWidget(self.empty, 1)
        self.no_match = widgets.label("No projects match that filter.", role="muted")
        self.no_match.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.no_match)

    def refresh(self, summaries: list[ProjectSummary], stats: HomeStats, active_model: ModelVersion | None,
                last_project: Path | None, processing: str | None = None, today: datetime | None = None) -> None:
        self._summaries = summaries
        self.greeting.setText(greeting_line(today or datetime.now(), stats))
        self.stat_matches.set(str(stats.matches_analysed), "Processed match videos")
        self.stat_cycles.set(str(stats.cycles), "Zone-trip cycles in analysed matches")
        self.stat_map.set(*model_stat(active_model))
        hours = stats.hours_processed
        self.stat_hours.set(f"{hours:.1f}" if hours >= 1 or hours == 0 else f"{hours:.2f}",
                            f"{hours * 60:.0f} minutes of match video" if 0 < hours < 1 else "Of match video")
        resumable = last_project is not None and Path(last_project).is_file()
        widgets.disable_with_reason(self.resume_btn, resumable, "No recent project")
        if processing:
            self.activity_text.setText(f"◐ {processing}")
        self.activity.setVisible(bool(processing))
        self._render()

    def visible_summaries(self) -> list[ProjectSummary]:
        season = self.season_filter.currentData()
        status = self.status_filter.currentData()
        return [summary for summary in self._summaries
                if not summary.archived
                and (season is None or summary.season == season)
                and (status is None or (not summary.missing and summary.status == status))]

    def _still(self, summary: ProjectSummary):
        path = summary.thumbnail_video
        if path is None:
            return None
        try:
            key = (str(path), Path(path).stat().st_mtime)
        except OSError:
            return None
        if key not in self._stills:
            self._stills[key] = widgets.video_still(path)
        return self._stills[key]

    def _render(self, *_):
        any_projects = any(not summary.archived for summary in self._summaries)
        visible = self.visible_summaries()
        cards = []
        for summary in visible:
            card = ProjectCard(summary, None if summary.missing else self._still(summary))
            card.clicked.connect(self.open_project.emit)
            cards.append(card)
        self.grid.set_widgets(cards)
        self.empty.setVisible(not any_projects)
        self.scroll.setVisible(any_projects)
        self.no_match.setVisible(any_projects and not visible)
        self.season_filter.setEnabled(any_projects)
        self.status_filter.setEnabled(any_projects)
