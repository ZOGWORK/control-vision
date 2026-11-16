"""Page 2 — Projects library: find a project fast, open, duplicate, archive or delete it."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMenu, QPushButton, QScrollArea, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from control_vision.app import widgets
from control_vision.app.library import PROJECT_STATUSES, ProjectSummary
from control_vision.app.pages.home import ProjectCard

COLUMNS = ("Project name", "Event code", "Matches", "Last opened", "Model version", "Accuracy", "Status")
PATH_ROLE = Qt.ItemDataRole.UserRole
SORT_ROLE = Qt.ItemDataRole.UserRole + 1


class SortItem(QTableWidgetItem):
    """Sorts by a hidden key (numbers, timestamps) instead of the displayed text."""

    def __lt__(self, other):
        mine, theirs = self.data(SORT_ROLE), other.data(SORT_ROLE)
        if mine is not None and theirs is not None:
            try:
                return mine < theirs
            except TypeError:
                pass
        return super().__lt__(other)


def model_cell(summary: ProjectSummary, active_name: str | None) -> str:
    """Version used for the last processing run; name the active one too when they differ."""
    used = summary.model_name
    if not used:
        return "—"
    if active_name and active_name != used:
        return f"{used}, active is {active_name}"
    return used


def last_opened_text(value: str | None) -> str:
    if not value:
        return "Never"
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        return value
    return f"{when:%b} {when.day}, {when:%Y} {when:%H:%M}"


def matches_filter(summary: ProjectSummary, text: str, season: str | None, event: str | None,
                   team: str, status: str | None, show_archived: bool) -> bool:
    """Search and filters combine: a project shows only when every one of them matches."""
    if summary.archived and not show_archived:
        return False
    text = text.strip().casefold()
    if text:
        haystack = " ".join(str(part) for part in (
            summary.display_name, summary.event_name or "", summary.event_code or "",
            " ".join(str(team_number) for team_number in sorted(summary.teams)))).casefold()
        if text not in haystack:
            return False
    if season and summary.season != season:
        return False
    if event and (summary.event_name or "") != event:
        return False
    if team.strip():
        if not team.strip().isdigit() or int(team.strip()) not in summary.teams:
            return False
    if status:
        if summary.missing or summary.status != status:
            return False
    return True


class ProjectsPage(QWidget):
    open_project = Signal(object)
    duplicate = Signal(object)
    archive = Signal(object, bool)
    delete = Signal(object)
    export_db = Signal(object)
    locate = Signal(object)
    forget = Signal(object)
    new_project = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        self._summaries: list[ProjectSummary] = []
        self._active_model: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search projects, events or team numbers")
        self.search.setAccessibleName("Search projects")
        self.search.setClearButtonEnabled(True)
        bar.addWidget(self.search, 2)
        self.season = QComboBox(); self.season.addItem("All seasons", None); self.season.addItem("BIOBUZZ 2026-27", "BIOBUZZ 2026-27")
        self.event = QComboBox(); self.event.addItem("All events", None)
        self.team = QLineEdit(); self.team.setPlaceholderText("Team #"); self.team.setMaximumWidth(90)
        self.status = QComboBox(); self.status.addItem("All statuses", None)
        for status in PROJECT_STATUSES:
            self.status.addItem(status, status)
        for widget, name in ((self.season, "Season filter"), (self.event, "Event filter"),
                             (self.team, "Team number filter"), (self.status, "Status filter")):
            widget.setAccessibleName(name)
            bar.addWidget(widget)
        self.show_archived = QCheckBox("Show &archived")
        bar.addWidget(self.show_archived)
        self.table_btn = widgets.button("Table"); self.grid_btn = widgets.button("Grid")
        self.view_group = QButtonGroup(self)
        for index, view_button in enumerate((self.table_btn, self.grid_btn)):
            view_button.setCheckable(True)
            view_button.setProperty("subnav", True)
            self.view_group.addButton(view_button, index)
            bar.addWidget(view_button)
        self.table_btn.setChecked(True)
        self.table_btn.setAccessibleName("Table view"); self.grid_btn.setAccessibleName("Grid view")
        layout.addLayout(bar)

        actions = QHBoxLayout()
        self.open_btn = widgets.button("Open", "primary", on_click=lambda: self._emit_selected(self.open_project))
        self.duplicate_btn = widgets.button("Duplicate", on_click=lambda: self._emit_selected(self.duplicate))
        self.archive_btn = widgets.button("Archive", on_click=self._archive_selected)
        self.export_btn = widgets.button("Export project .db", on_click=lambda: self._emit_selected(self.export_db))
        self.locate_btn = widgets.button("Locate file…", on_click=lambda: self._emit_selected(self.locate))
        self.delete_btn = widgets.button("Delete…", "danger", on_click=lambda: self._emit_selected(self.delete))
        for action in (self.open_btn, self.duplicate_btn, self.archive_btn, self.export_btn, self.locate_btn):
            actions.addWidget(action)
        actions.addStretch(1)
        actions.addWidget(self.delete_btn)
        layout.addLayout(actions)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setAccessibleName("Projects")
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSortingEnabled(True)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._row_menu)
        self.table.cellDoubleClicked.connect(lambda *_: self._emit_selected(self.open_project))
        self.table.itemSelectionChanged.connect(self._sync_actions)

        self.grid = widgets.FlowGrid(3)
        grid_holder = QWidget(); grid_layout = QVBoxLayout(grid_holder)
        grid_layout.setContentsMargins(0, 0, 0, 0); grid_layout.addWidget(self.grid); grid_layout.addStretch(1)
        grid_scroll = QScrollArea(); grid_scroll.setObjectName("PageScroll"); grid_scroll.setWidgetResizable(True)
        grid_scroll.setWidget(grid_holder)

        self.views = QStackedWidget()
        self.views.addWidget(self.table)
        self.views.addWidget(grid_scroll)
        layout.addWidget(self.views, 1)

        self.no_projects = widgets.EmptyState("No projects yet", "Create a project to see it here.",
                                              "+ New project", self.new_project.emit)
        self.no_match = widgets.EmptyState("No projects match that filter", "", "Clear filters",
                                           self.clear_filters, primary=False)
        layout.addWidget(self.no_projects, 1)
        layout.addWidget(self.no_match, 1)
        self.count = widgets.label("", role="muted")
        self.count.setAccessibleName("Project count")
        layout.addWidget(self.count)

        self.search.textChanged.connect(self._render)
        self.team.textChanged.connect(self._render)
        for combo in (self.season, self.event, self.status):
            combo.currentIndexChanged.connect(self._render)
        self.show_archived.toggled.connect(self._render)
        self.view_group.idClicked.connect(self.views.setCurrentIndex)
        self._sync_actions()

    # ----- data -----

    def refresh(self, summaries: list[ProjectSummary], active_model: str | None) -> None:
        self._summaries = summaries
        self._active_model = active_model
        current = self.event.currentData()
        self.event.blockSignals(True)
        self.event.clear()
        self.event.addItem("All events", None)
        for name in sorted({summary.event_name for summary in summaries if summary.event_name}):
            self.event.addItem(name, name)
        index = self.event.findData(current)
        self.event.setCurrentIndex(index if index >= 0 else 0)
        self.event.blockSignals(False)
        self._render()

    def visible(self) -> list[ProjectSummary]:
        return [summary for summary in self._summaries if matches_filter(
            summary, self.search.text(), self.season.currentData(), self.event.currentData(),
            self.team.text(), self.status.currentData(), self.show_archived.isChecked())]

    def clear_filters(self) -> None:
        for widget in (self.search, self.team):
            widget.clear()
        for combo in (self.season, self.event, self.status):
            combo.setCurrentIndex(0)

    def _render(self, *_):
        visible = self.visible()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(visible))
        for row, summary in enumerate(visible):
            if summary.missing:
                status_text, status_key = "✕ Missing file — Locate or remove", "Missing file"
            elif summary.unreadable:
                status_text, status_key = f"✕ Cannot read: {summary.unreadable}", "Unreadable"
            else:
                status_text = widgets.PROJECT_STATUS_BADGES.get(summary.status, ("", summary.status))[1]
                status_key = summary.status
                if summary.archived:
                    status_text += " · Archived"
            name = summary.display_name if not summary.missing else f"{summary.display_name} ({summary.path.name})"
            cells = (
                (name, name.casefold()),
                (summary.event_code or "—", summary.event_code or ""),
                (str(summary.matches), summary.matches),
                (last_opened_text(summary.last_opened), summary.last_opened or ""),
                (model_cell(summary, self._active_model), summary.model_name or ""),
                ("—", ""),  # Accuracy appears only with a measured holdout report.
                (status_text, status_key),
            )
            for column, (text, key) in enumerate(cells):
                item = SortItem(text)
                item.setData(SORT_ROLE, key)
                item.setData(PATH_ROLE, str(summary.path))
                if column == 0:
                    item.setToolTip(str(summary.path))
                self.table.setItem(row, column, item)
        self.table.setSortingEnabled(True)
        cards = []
        for summary in visible:
            card = ProjectCard(summary)
            card.clicked.connect(self.open_project.emit)
            cards.append(card)
        self.grid.set_widgets(cards)
        any_projects = bool(self._summaries)
        self.views.setVisible(bool(visible))
        self.no_projects.setVisible(not any_projects)
        self.no_match.setVisible(any_projects and not visible)
        archived = sum(1 for summary in self._summaries if summary.archived)
        total = len(self._summaries) - archived
        self.count.setText(f"{total} project" + ("" if total == 1 else "s") + f", {archived} archived")
        self._sync_actions()

    # ----- selection and actions -----

    def selected(self) -> ProjectSummary | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        path = self.table.item(rows[0].row(), 0).data(PATH_ROLE)
        return next((summary for summary in self._summaries if str(summary.path) == path), None)

    def _emit_selected(self, signal):
        summary = self.selected()
        if summary is not None:
            signal.emit(summary.path)

    def _archive_selected(self):
        summary = self.selected()
        if summary is not None:
            self.archive.emit(summary.path, not summary.archived)

    def _sync_actions(self):
        summary = self.selected()
        usable = summary is not None and not summary.missing and not summary.unreadable
        for control, enabled, reason in (
                (self.open_btn, usable, "Select a project that is on this computer"),
                (self.duplicate_btn, usable, "Select a project that is on this computer"),
                (self.archive_btn, usable, "Select a project that is on this computer"),
                (self.export_btn, usable, "Select a project that is on this computer"),
                (self.locate_btn, summary is not None and summary.missing, "Only for a project whose file is missing"),
                (self.delete_btn, summary is not None, "Select a project")):
            widgets.disable_with_reason(control, enabled, reason)
        self.archive_btn.setText("Unarchive" if summary is not None and summary.archived else "Archive")
        self.delete_btn.setText("Remove from list" if summary is not None and summary.missing else "Delete…")

    def _row_menu(self, position):
        if self.table.itemAt(position) is None:
            return
        self.table.selectRow(self.table.itemAt(position).row())
        summary = self.selected()
        menu = QMenu(self)
        if summary.missing:
            menu.addAction("Locate file…", lambda: self.locate.emit(summary.path))
            menu.addAction("Remove from list", lambda: self.forget.emit(summary.path))
        else:
            menu.addAction("Open", lambda: self.open_project.emit(summary.path))
            menu.addAction("Duplicate", lambda: self.duplicate.emit(summary.path))
            menu.addAction("Unarchive" if summary.archived else "Archive",
                           lambda: self.archive.emit(summary.path, not summary.archived))
            menu.addAction("Export project .db…", lambda: self.export_db.emit(summary.path))
            menu.addSeparator()
            menu.addAction("Delete…", lambda: self.delete.emit(summary.path))
        menu.exec(self.table.viewport().mapToGlobal(position))
