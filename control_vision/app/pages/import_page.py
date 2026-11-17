"""Page 4 — Import videos: get match videos into the project and check they are usable first."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLineEdit, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from control_vision.app import theme, widgets
from control_vision.app.media import FileCheck, FileChecker, SUPPORTED_TEXT, format_clock
from control_vision.app.pages.wizard import YOUTUBE

CAMERA_VIEWS = (("Unknown", "unknown"), ("Wide", "wide"), ("Close", "close"))
QUEUE_COLUMNS = ("File", "Duration", "Resolution", "Fps", "Size", "Camera view", "Match", "Holdout", "Status")
PROJECT_COLUMNS = ("File", "Match", "Duration", "Resolution", "Camera view", "Holdout", "Status")
PATH_ROLE = Qt.ItemDataRole.UserRole + 1


@dataclass
class QueuedFile:
    path: Path
    check: FileCheck | None = None  # None while it is being checked
    camera_view: str = "unknown"
    match: str = ""
    holdout: bool = False
    import_anyway: bool = False


def is_duplicate(check: FileCheck, known: list[tuple[str, float]]) -> bool:
    """Same file name and the same duration (within half a second) as a known video."""
    if not check.ok:
        return False
    name, duration = check.path.name.casefold(), check.duration_s
    return any(name == other_name and abs(duration - other_duration) <= 0.5 for other_name, other_duration in known)


def queue_status(item: QueuedFile, duplicate: bool) -> tuple[str, str]:
    """(text, tone) for the Status column; colour always comes with words."""
    check = item.check
    if check is None:
        return "◐ Checking…", "info"
    if check.error:
        return f"✕ {check.error}", "danger"
    if duplicate and not item.import_anyway:
        return "○ Duplicate — skipped (Import anyway to keep it)", "muted"
    notes = []
    if check.long:
        notes.append("Long video — split after import")
    if check.vertical:
        notes.append("Vertical — field may be cropped")
    if check.audio is False:
        notes.append("No audio, OK")
    prefix = "⚑ " if check.long or check.vertical or check.warnings else "✓ Ready"
    if prefix == "✓ Ready":
        return prefix + (f" · {notes[0]}" if notes else ""), "success"
    return prefix + " · ".join(notes or ["Ready, with warnings"]), "pending"


def size_text(size: int | None) -> str:
    if size is None:
        return "—"
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    return f"{size / 1_000_000:.1f} MB"


class ImportPage(QWidget):
    imported = Signal(list)          # ids of the new videos
    open_splitter = Signal(object)   # path of a long video
    open_calibrate = Signal(int)
    open_identity = Signal(int)
    new_project = Signal()

    def __init__(self, checker: FileChecker | None = None):
        super().__init__()
        self.setObjectName("Page")
        self.db = None
        self.project_id = None
        self.queue: list[QueuedFile] = []
        self.checker = checker or FileChecker(self)
        self.checker.checked.connect(self._checked)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        self.no_project = widgets.EmptyState("Open a project to import videos",
                                             "Videos are added to a project. Create one, or open one from Projects.",
                                             "+ New project", self.new_project.emit)
        outer.addWidget(self.no_project, 1)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        outer.addWidget(self.body, 1)

        drop = widgets.card()
        drop_layout = QHBoxLayout(drop)
        drop_layout.setContentsMargins(16, 12, 16, 12)
        hint = QVBoxLayout()
        hint.addWidget(widgets.label("Drop videos here, or Browse", role="section"))
        hint.addWidget(widgets.label(f"{SUPPORTED_TEXT}. Videos stay where they are; the project stores their paths.",
                                     role="muted"))
        drop_layout.addLayout(hint, 1)
        self.browse_btn = widgets.button("Browse…", "primary", on_click=self.browse)
        drop_layout.addWidget(self.browse_btn)
        self.youtube = QLineEdit(); self.youtube.setPlaceholderText("Or paste a YouTube link")
        self.youtube.setAccessibleName("YouTube link"); self.youtube.setFixedWidth(260)
        drop_layout.addWidget(self.youtube)
        drop_layout.addWidget(widgets.button("Check link", on_click=self.check_link))
        body.addWidget(drop)
        self.youtube_status = widgets.label("", wrap=True)
        self.youtube_status.hide()
        body.addWidget(self.youtube_status)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        left = QWidget(); left_layout = QVBoxLayout(left); left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(widgets.section_label("To import"))
        self.table = DropTable(len(QUEUE_COLUMNS))
        self.table.setHorizontalHeaderLabels(QUEUE_COLUMNS)
        self.table.setAccessibleName("Videos to import")
        self.table.files_dropped.connect(self.add_paths)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        left_layout.addWidget(self.table, 1)
        self.queue_empty = widgets.label("Nothing to import yet. Drop files above or press Browse.", role="muted")
        left_layout.addWidget(self.queue_empty)
        actions = QHBoxLayout()
        self.remove_btn = widgets.button("Remove selected", on_click=self.remove_selected)
        self.holdout_btn = widgets.button("Mark as holdout", on_click=self.toggle_holdout)
        self.anyway_btn = widgets.button("Import anyway", on_click=self.import_anyway)
        self.count = widgets.label("", role="muted")
        self.import_btn = widgets.button("Import", "primary", on_click=self.import_ready)
        for control in (self.remove_btn, self.holdout_btn, self.anyway_btn):
            actions.addWidget(control)
        actions.addStretch(1)
        actions.addWidget(self.count)
        actions.addWidget(self.import_btn)
        left_layout.addLayout(actions)
        split.addWidget(left)

        preview = widgets.card()
        preview.setMinimumWidth(300)
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.addWidget(widgets.section_label("Preview"))
        self.preview = widgets.Thumbnail(288)
        self.preview.show_placeholder("Select a video to preview it")
        preview_layout.addWidget(self.preview)
        self.preview_info = widgets.label("", role="muted", wrap=True)
        preview_layout.addWidget(self.preview_info)
        preview_layout.addStretch(1)
        split.addWidget(preview)
        split.setStretchFactor(0, 3)
        body.addWidget(split, 3)

        body.addWidget(widgets.section_label("In this project"))
        self.project_table = QTableWidget(0, len(PROJECT_COLUMNS))
        self.project_table.setHorizontalHeaderLabels(PROJECT_COLUMNS)
        self.project_table.setAccessibleName("Videos in this project")
        self.project_table.itemChanged.connect(self._project_item_changed)
        self.project_table.itemSelectionChanged.connect(self._sync_project_actions)
        for table in (self.table, self.project_table):
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.verticalHeader().setVisible(False)
            table.setShowGrid(False)
            table.setAlternatingRowColors(True)
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setMinimumSectionSize(60)
        # Queue: the file name and its status share the spare width; status text is in a tooltip too.
        queue_header = self.table.horizontalHeader()
        queue_header.setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        for column, width in ((5, 130), (6, 90), (7, 100)):
            queue_header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        self.table.setIconSize(QPixmap(64, 36).size())
        body.addWidget(self.project_table, 2)
        project_actions = QHBoxLayout()
        self.calibrate_btn = widgets.button("Calibrate…", on_click=lambda: self._emit_project(self.open_calibrate))
        self.identity_btn = widgets.button("Assign robots…", on_click=lambda: self._emit_project(self.open_identity))
        self.split_btn = widgets.button("Split this stream…", on_click=self._split_selected)
        for control in (self.calibrate_btn, self.identity_btn, self.split_btn):
            project_actions.addWidget(control)
        project_actions.addStretch(1)
        body.addLayout(project_actions)
        self.set_project(None, None)

    # ----- project -----

    def set_project(self, db, project_id) -> None:
        if db is not self.db or project_id != self.project_id:
            self.queue = []  # Files picked for one project are not imported into another.
        self.db, self.project_id = db, project_id
        has_project = db is not None and project_id is not None
        self.no_project.setVisible(not has_project)
        self.body.setVisible(has_project)
        self.refresh_project()
        self._render_queue()

    def project_videos(self) -> list:
        if self.db is None or self.project_id is None:
            return []
        return self.db.connection.execute(
            "SELECT v.*, (SELECT COUNT(*) FROM calibrations c WHERE c.video_id=v.id) AS calibrated, "
            "(SELECT COUNT(*) FROM robots r WHERE r.video_id=v.id) AS robot_count "
            "FROM videos v WHERE v.project_id=? ORDER BY v.id", (self.project_id,)).fetchall()

    def refresh_project(self) -> None:
        rows = self.project_videos()
        table = self.project_table
        table.blockSignals(True)
        table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            path = Path(row["path"])
            if not path.is_file():
                status, tone = "✕ File missing", "danger"
            elif row["analysis_status"] == "ready":
                status, tone = "✓ Processed", "success"
            elif not row["calibrated"]:
                status, tone = "⚑ Needs calibration", "pending"
            elif not row["robot_count"]:
                status, tone = "⚑ Needs robot identities", "pending"
            else:
                status, tone = "○ Ready to process", "muted"
            cells = (path.name, row["match_number"] or "—", format_clock(row["duration_s"] or 0),
                     f"{row['width']}x{row['height']}", None, None, status)
            for column, text in enumerate(cells):
                if text is None:
                    continue
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, int(row["id"]))
                item.setData(PATH_ROLE, str(path))
                if column == 0:
                    item.setToolTip(str(path))
                if column == 6:
                    item.setForeground(_tone_brush(tone))
                table.setItem(index, column, item)
            holdout = QTableWidgetItem("⚑ Holdout" if row["is_holdout"] else "")
            holdout.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            holdout.setCheckState(Qt.CheckState.Checked if row["is_holdout"] else Qt.CheckState.Unchecked)
            holdout.setData(Qt.ItemDataRole.UserRole, int(row["id"]))
            holdout.setToolTip("Holdout videos are kept out of training and used to measure accuracy.")
            if row["is_holdout"]:
                holdout.setForeground(_tone_brush("pending"))
            table.setItem(index, 5, holdout)
            video_id = int(row["id"])
            combo = cell_widget(table, index, 4, f"video:{video_id}",
                                lambda vid=video_id: self._camera_combo("unknown", lambda view: self._set_view(vid, view)))
            set_combo_value(combo, row["camera_view"] or "unknown")
        table.blockSignals(False)
        self._sync_project_actions()

    def _set_view(self, video_id: int, view: str) -> None:
        if self.db is not None:
            self.db.set_video_details(video_id, camera_view=view)

    def _project_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 5 or self.db is None:
            return
        self.db.set_video_details(item.data(Qt.ItemDataRole.UserRole), is_holdout=item.checkState() == Qt.CheckState.Checked)
        self.refresh_project()

    def selected_project_video(self) -> tuple[int, Path] | None:
        rows = self.project_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.project_table.item(rows[0].row(), 0)
        return int(item.data(Qt.ItemDataRole.UserRole)), Path(item.data(PATH_ROLE))

    def _emit_project(self, signal) -> None:
        selected = self.selected_project_video()
        if selected is not None:
            signal.emit(selected[0])

    def _split_selected(self) -> None:
        selected = self.selected_project_video()
        if selected is not None:
            self.open_splitter.emit(selected[1])

    def _sync_project_actions(self) -> None:
        selected = self.selected_project_video()
        usable = selected is not None and selected[1].is_file()
        for control in (self.calibrate_btn, self.identity_btn, self.split_btn):
            widgets.disable_with_reason(control, usable, "Select a video whose file is on this computer")

    # ----- queue -----

    def browse(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add match videos", "", "Videos (*.mp4 *.mov *.mkv);;All files (*)")
        if files:
            self.add_paths([Path(file) for file in files])

    def add_paths(self, paths) -> None:
        if self.db is None:
            return
        known = {item.path.resolve() for item in self.queue}
        for path in paths:
            path = Path(path)
            if path.resolve() in known:
                continue
            known.add(path.resolve())
            self.queue.append(QueuedFile(path=path))
            self.checker.check(path)
        self._render_queue()

    def _checked(self, check: FileCheck) -> None:
        for item in self.queue:
            if item.path.resolve() == check.path.resolve() and item.check is None:
                item.check = check
        self._render_queue()

    def _known_videos(self, before: int) -> list[tuple[str, float]]:
        known = [(Path(row["path"]).name.casefold(), float(row["duration_s"] or 0)) for row in self.project_videos()]
        known += [(item.path.name.casefold(), item.check.duration_s) for item in self.queue[:before]
                  if item.check is not None and item.check.ok]
        return known

    def duplicates(self) -> list[bool]:
        return [item.check is not None and is_duplicate(item.check, self._known_videos(index))
                for index, item in enumerate(self.queue)]

    def ready(self) -> list[QueuedFile]:
        return [item for item, duplicate in zip(self.queue, self.duplicates())
                if item.check is not None and item.check.ok and (not duplicate or item.import_anyway)]

    def _render_queue(self) -> None:
        table = self.table
        selected = {index.row() for index in table.selectionModel().selectedRows()}
        table.blockSignals(True)
        table.setRowCount(len(self.queue))
        for row, (item, duplicate) in enumerate(zip(self.queue, self.duplicates())):
            check = item.check
            status, tone = queue_status(item, duplicate)
            values = (item.path.name,
                      format_clock(check.duration_s) if check and check.ok else "—",
                      check.resolution if check and check.ok else "—",
                      f"{check.metadata['fps']:.2f}" if check and check.ok else "—",
                      size_text(check.size if check else None))
            for column, text in enumerate(values):
                cell = QTableWidgetItem(text)
                if column == 0:
                    cell.setToolTip(str(item.path))
                table.setItem(row, column, cell)
            status_item = QTableWidgetItem(status)
            status_item.setForeground(_tone_brush(tone))
            status_item.setToolTip("\n".join([status] + (list(check.warnings) if check is not None else [])))
            table.setItem(row, 8, status_item)
            key = str(item.path.resolve())
            combo = cell_widget(table, row, 5, key, lambda i=item: self._camera_combo(
                i.camera_view, lambda view: setattr(i, "camera_view", view)))
            set_combo_value(combo, item.camera_view)
            match = cell_widget(table, row, 6, key, lambda i=item: self._match_edit(i))
            if not match.hasFocus() and match.text().strip() != item.match:
                match.setText(item.match)
            holdout = cell_widget(table, row, 7, key, lambda i=item: self._holdout_box(i))
            if holdout.isChecked() != item.holdout:
                holdout.blockSignals(True)
                holdout.setChecked(item.holdout)
                holdout.blockSignals(False)
        for row in selected:
            if row < table.rowCount():
                table.selectRow(row)
        table.blockSignals(False)
        self.queue_empty.setVisible(not self.queue)
        table.setVisible(bool(self.queue))
        self._update_count()
        self._selection_changed()

    def _match_edit(self, item: QueuedFile) -> QLineEdit:
        match = QLineEdit(item.match)
        match.setPlaceholderText("e.g. Q12")
        match.setAccessibleName(f"Match number for {item.path.name}")
        match.textChanged.connect(lambda text: setattr(item, "match", text.strip()))
        return match

    def _holdout_box(self, item: QueuedFile) -> QCheckBox:
        holdout = QCheckBox("Holdout")
        holdout.setChecked(item.holdout)
        holdout.setAccessibleName(f"Holdout for {item.path.name}")
        holdout.toggled.connect(lambda on: (setattr(item, "holdout", on), self._update_count()))
        return holdout

    def _camera_combo(self, current: str, on_change) -> QComboBox:
        combo = QComboBox()
        for text, value in CAMERA_VIEWS:
            combo.addItem(text, value)
        combo.setCurrentIndex(max(0, combo.findData(current)))
        combo.setAccessibleName("Camera view")
        combo.setToolTip("Set by a person who has looked at the video; never guessed.")
        combo.currentIndexChanged.connect(lambda _index: on_change(combo.currentData()))
        return combo

    def _update_count(self) -> None:
        ready = self.ready()
        checking = sum(1 for item in self.queue if item.check is None)
        text = f"{len(ready)} of {len(self.queue)} ready"
        if checking:
            text += f", {checking} checking"
        self.count.setText(text)
        reason = "Wait for the checks to finish" if checking else "No video is ready to import"
        widgets.disable_with_reason(self.import_btn, bool(ready) and not checking, reason)
        self.import_btn.setText(f"Import {len(ready)} video" + ("" if len(ready) == 1 else "s") if ready else "Import")

    def _selected_rows(self) -> list[int]:
        return sorted(index.row() for index in self.table.selectionModel().selectedRows())

    def _selection_changed(self) -> None:
        rows = self._selected_rows()
        has = bool(rows)
        widgets.disable_with_reason(self.remove_btn, has, "Select a video in the list")
        widgets.disable_with_reason(self.holdout_btn, has, "Select a video in the list")
        duplicates = self.duplicates()
        any_duplicate = any(0 <= row < len(self.queue) and duplicates[row] for row in rows)
        widgets.disable_with_reason(self.anyway_btn, any_duplicate, "Only for a video flagged as a duplicate")
        if rows and rows[0] < len(self.queue):
            self._show_preview(self.queue[rows[0]])

    def _show_preview(self, item: QueuedFile) -> None:
        if item.check is None or not item.check.ok:
            self.preview.show_placeholder(item.path.name)
            self.preview_info.setText(item.check.error if item.check else "Checking…")
            return
        still = widgets.video_still(item.path, 0.1)
        if still is not None:
            self.preview.show_image(still)
        else:
            self.preview.show_placeholder(item.path.name)
        check = item.check
        audio = {True: "Has audio", False: "No audio, OK", None: "Audio not checked"}[check.audio]
        self.preview_info.setText(f"{item.path.name}\n{format_clock(check.duration_s)} · {check.resolution} · "
                                  f"{check.metadata['fps']:.2f} fps · {size_text(check.size)} · {audio}")

    def remove_selected(self) -> None:
        for row in reversed(self._selected_rows()):
            if 0 <= row < len(self.queue):
                del self.queue[row]  # Only taken off the list; the file is untouched.
        self._render_queue()

    def toggle_holdout(self) -> None:
        rows = [row for row in self._selected_rows() if row < len(self.queue)]
        mark = not all(self.queue[row].holdout for row in rows)
        for row in rows:
            self.queue[row].holdout = mark
        self._render_queue()

    def import_anyway(self) -> None:
        duplicates = self.duplicates()
        for row in self._selected_rows():
            if row < len(self.queue) and duplicates[row]:
                self.queue[row].import_anyway = True
        self._render_queue()

    def import_ready(self) -> list[int]:
        """Add every ready video to the project (paths only) and take them off the list."""
        if self.db is None or self.project_id is None:
            return []
        details = self.db.project_details(self.project_id) or {}
        ids, done = [], []
        for item in self.ready():
            check = item.check
            video_id = self.db.add_video(self.project_id, item.path, check.metadata, details.get("event_code"),
                                         item.match or None)
            self.db.set_video_details(video_id, is_holdout=item.holdout, camera_view=item.camera_view, file_size=check.size)
            ids.append(video_id)
            done.append(item)
        self.queue = [item for item in self.queue if item not in done]
        self._render_queue()
        self.refresh_project()
        if ids:
            self.imported.emit(ids)
        return ids

    def check_link(self) -> None:
        link = self.youtube.text().strip()
        self.youtube_status.show()
        if not YOUTUBE.match(link):
            theme.set_tone(self.youtube_status, "danger")
            self.youtube_status.setText("✕ That link did not load, check it and try again")
            return
        theme.set_tone(self.youtube_status, "pending")
        self.youtube_status.setText(f"⚠ {link}\nDownloading from YouTube is not available in this build, and nothing "
                                    "is downloaded without your OK. Save the video as an MP4, then add the file.")


class DropTable(QTableWidget):
    """A table that accepts dropped files, like the wizard's list."""

    files_dropped = Signal(list)

    def __init__(self, columns: int):
        super().__init__(0, columns)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


def cell_widget(table, row: int, column: int, key: str, create):
    """The widget in a cell, reused while it still belongs to the same file or video.

    Replacing a cell widget on every refresh leaves the old one on screen until Qt deletes it
    later, and would throw away a field the user is typing in when a background check finishes.
    """
    widget = table.cellWidget(row, column)
    if widget is not None and widget.property("cell_key") == key:
        return widget
    if widget is not None:
        widget.hide()
    widget = create()
    widget.setProperty("cell_key", key)
    table.setCellWidget(row, column, widget)
    return widget


def set_combo_value(combo: QComboBox, value: str) -> None:
    index = max(0, combo.findData(value))
    if combo.currentIndex() != index:
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)


def _tone_brush(tone: str):
    from PySide6.QtGui import QBrush, QColor
    from PySide6.QtWidgets import QApplication
    tokens = theme.resolve_tokens(QApplication.instance().property("control_vision_theme") or theme.DEFAULT_MODE)
    colour = {"success": tokens.success_fg, "pending": tokens.pending_fg, "danger": tokens.danger_fg,
              "muted": tokens.text_muted}.get(tone, tokens.text)
    return QBrush(QColor(colour))
