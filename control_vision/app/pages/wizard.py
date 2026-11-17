"""Page 3 — New project wizard: four steps, nothing saved until Create Project."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtWidgets import (
    QDateEdit, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPlainTextEdit, QStackedWidget, QVBoxLayout, QWidget,
)

from control_vision.app import theme, widgets
from control_vision.app.library import MODEL_CLASSES, SEASON, ModelStore, ModelVersion, now_iso, unique_name
from control_vision.storage.database import normalise_event_code

STEPS = ("Name and event", "Add videos", "Choose model", "Confirm")
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv")
YOUTUBE = re.compile(r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|live/|shorts/)|youtu\.be/)[\w-]{6,}", re.I)


@dataclass
class WizardVideo:
    path: Path
    duration_s: float | None = None
    size: int | None = None
    metadata: dict | None = None
    error: str | None = None

    @property
    def usable(self) -> bool:
        """Readable when it was added and still there now."""
        return not self.error and bool(self.metadata) and self.path.is_file()

    def describe(self) -> str:
        parts = [self.path.name]
        if self.duration_s is not None:
            parts.append(f"{int(self.duration_s // 60)}:{int(self.duration_s % 60):02d}")
        if self.size is not None:
            parts.append(f"{self.size / 1_000_000:.1f} MB")
        if self.error:
            parts.append(f"✕ {self.error}")
        return "  ·  ".join(parts)


@dataclass
class ProjectSpec:
    name: str
    event_name: str
    event_code: str | None
    event_date: str
    notes: str
    folder: Path | None
    videos: list[WizardVideo] = field(default_factory=list)
    model: ModelVersion | None = None


def check_video(path: Path) -> WizardVideo:
    """Read duration and size from the file itself; never estimate them."""
    path = Path(path)
    video = WizardVideo(path=path)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        video.error = "That file type is not supported (MP4, MOV, MKV)"
        return video
    try:
        video.size = path.stat().st_size
    except OSError as exc:
        video.error = f"Cannot read the file: {exc}"
        return video
    try:
        from control_vision.pipeline.ingest import probe_video
        video.metadata = probe_video(path)
        video.duration_s = float(video.metadata["duration_s"])
    except Exception as exc:  # A file that cannot be probed is shown, not imported.
        video.error = f"Cannot read this video: {exc}"
    return video


def step_problem(step: int, spec: ProjectSpec, existing_names: set[str], models: list[ModelVersion]) -> str | None:
    """Why Next (or Create) is disabled on ``step``, or None when the step is valid."""
    if step == 0:
        if not spec.name.strip():
            return "Enter a project name"
        if spec.name.strip().casefold() in {name.casefold() for name in existing_names}:
            return f"A project with that name exists — try \"{unique_name(spec.name, existing_names)}\""
        if not spec.event_name.strip():
            return "Enter an event name"
        if spec.folder is None:
            return "Choose a project folder"
        if (Path(spec.folder) / "control_vision.sqlite").exists():
            return "That folder already holds a project — choose another"
        return None
    if step == 1:
        if not [video for video in spec.videos if video.usable]:
            return "Add at least 1 video"
        return None
    if step == 2:
        if not models:
            return "No models yet — train one first"
        if spec.model is None:
            return "Pick a model"
        if not any(model.id == spec.model.id for model in models) or not Path(spec.model.path).is_file():
            return "That model is gone, pick another"
        return None
    return None


def create_project(spec: ProjectSpec) -> Path:
    """Write the project: folder, database, details, video paths (never copies) and model choice."""
    from control_vision.storage.database import ProjectDatabase

    from control_vision.app.library import remove_partial_database

    folder = Path(spec.folder)
    db_path = folder / "control_vision.sqlite"
    if db_path.exists():
        raise FileExistsError(f"{db_path} already exists.")
    created_folder = not folder.exists()
    run_logs, settings_path = folder / "run_logs", folder / "settings.json"
    created_logs, created_settings = not run_logs.exists(), not settings_path.exists()
    database = None
    try:
        # The side files first, so a folder that cannot take them never gets a database.
        folder.mkdir(parents=True, exist_ok=True)
        run_logs.mkdir(exist_ok=True)
        settings = {"model_path": spec.model.path if spec.model else "", "tracking_mode": "yolo"}
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        database = ProjectDatabase(db_path)
        project_id = database.add_project(spec.name.strip())
        database.set_project_details(
            project_id, event_name=spec.event_name.strip(),
            event_code=normalise_event_code(spec.event_code) if spec.event_code else None,
            event_date=spec.event_date, season=SEASON, notes=spec.notes.strip() or None,
            model_name=spec.model.name if spec.model else None,
            model_path=spec.model.path if spec.model else None, last_opened=now_iso())
        code = normalise_event_code(spec.event_code) if spec.event_code else None
        for video in spec.videos:
            if not video.usable:
                continue
            video_id = database.add_video(project_id, video.path, video.metadata, code, None)
            database.set_video_details(video_id, file_size=video.size)
        database.close()
        database = None
    except BaseException:
        # Leave the folder as it was, so the same folder can be used on a retry.
        if database is not None:
            database.close()
        remove_partial_database(db_path)
        for path, created in ((settings_path, created_settings),):
            if created:
                path.unlink(missing_ok=True)
        for path, created in ((run_logs, created_logs), (folder, created_folder)):
            if created:
                try:
                    path.rmdir()
                except OSError:
                    pass
        raise
    return db_path


def model_row_text(model: ModelVersion, recommended: bool) -> str:
    def metric(value):
        return f"{value:.2f}" if value is not None else "—"
    classes = ", ".join(f"{name} {metric(model.class_metric(name, 'map50'))}" for name in MODEL_CLASSES)
    marker = "● Recommended  " if recommended else ""
    missing = "  ·  ✕ File missing" if not model.file_exists else ""
    return (f"{marker}{model.name}  ·  {model.status}{missing}\n"
            f"mAP50 {metric(model.map50)}  ·  mAP50-95 {metric(model.map50_95)}  ·  per class: {classes}")


class DropList(QListWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

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


class NewProjectWizard(QWidget):
    created = Signal(object)        # path of the new project database
    cancelled = Signal()
    open_models = Signal()

    def __init__(self, model_store: ModelStore | None = None, existing_names=lambda: set(), default_folder=None):
        super().__init__()
        self.setObjectName("Page")
        self.model_store = model_store or ModelStore()
        self.existing_names = existing_names
        self.default_folder = default_folder or (lambda name: Path.home() / "Documents" / "Control Vision" / name)
        self.step = 0
        self.videos: list[WizardVideo] = []
        self._folder_edited = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        steps_card = widgets.card()
        steps_card.setFixedWidth(230)
        steps_layout = QVBoxLayout(steps_card)
        steps_layout.setContentsMargins(16, 16, 16, 16)
        steps_layout.addWidget(widgets.section_label("New project"))
        self.step_labels = []
        for index, title in enumerate(STEPS):
            row = QHBoxLayout()
            number = widgets.label(str(index + 1), role="step-number")
            text = widgets.label(title)
            row.addWidget(number); row.addWidget(text, 1)
            steps_layout.addLayout(row)
            self.step_labels.append((number, text))
        steps_layout.addStretch(1)
        layout.addWidget(steps_card)

        main = QVBoxLayout()
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_details())
        self.pages.addWidget(self._build_videos())
        self.pages.addWidget(self._build_models())
        self.pages.addWidget(self._build_confirm())
        main.addWidget(self.pages, 1)
        footer = QHBoxLayout()
        self.cancel_btn = widgets.button("Cancel", on_click=self.cancel)
        self.back_btn = widgets.button("‹ Back", on_click=self.back)
        self.reason = widgets.label("", role="muted")
        self.reason.setAccessibleName("Why Next is disabled")
        self.next_btn = widgets.button("Next ›", "primary", on_click=self.next)
        self.create_btn = widgets.button("Create project", "primary", on_click=self.create)
        footer.addWidget(self.cancel_btn)
        footer.addStretch(1)
        footer.addWidget(self.reason)
        footer.addWidget(self.back_btn)
        footer.addWidget(self.next_btn)
        footer.addWidget(self.create_btn)
        main.addLayout(footer)
        layout.addLayout(main, 1)
        self.error = widgets.label("", tone="danger", wrap=True)
        self.error.hide()
        main.insertWidget(1, self.error)
        self.reset()

    # ----- steps -----

    def _build_details(self) -> QWidget:
        page = widgets.card()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(20, 20, 20, 20)
        form = QFormLayout()
        outer.addLayout(form)
        outer.addStretch(1)  # Spare height goes below the form, not into a row.
        form.setVerticalSpacing(12)
        self.name = QLineEdit(); self.name.setPlaceholderText("e.g. Salk League Meet 1")
        self.event_name = QLineEdit(); self.event_name.setPlaceholderText("Event name, as on the schedule")
        self.event_code = QLineEdit(); self.event_code.setPlaceholderText("Optional — only if you know it")
        self.date = QDateEdit(QDate.currentDate()); self.date.setCalendarPopup(True); self.date.setDisplayFormat("yyyy-MM-dd")
        self.season = widgets.label(f"{SEASON} (locked)", role="muted")
        self.notes = QPlainTextEdit(); self.notes.setPlaceholderText("Optional"); self.notes.setMaximumHeight(90)
        self.folder = QLineEdit()
        folder_row = QHBoxLayout(); folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(widgets.button("Browse…", on_click=self._browse_folder))
        self.name_hint = widgets.label("", role="muted")
        self.use_suggestion = widgets.button("", "link", on_click=self._use_suggestion)
        self.use_suggestion.hide()
        name_row = QHBoxLayout(); name_row.addWidget(self.name, 1); name_row.addWidget(self.use_suggestion)
        for text, widget, accessible in (
                ("Project &name *", name_row, "Project name"), ("&Event name *", self.event_name, "Event name"),
                ("Event &code", self.event_code, "Event code"), ("&Date", self.date, "Date"),
                ("Season", self.season, "Season"), ("N&otes", self.notes, "Notes"),
                ("Project &folder *", folder_row, "Project folder")):
            caption = QLabel(text)
            target = widget if isinstance(widget, QWidget) else None
            if target is None:
                target = {"Project name": self.name, "Project folder": self.folder}[accessible]
            caption.setBuddy(target)
            target.setAccessibleName(accessible)
            form.addRow(caption, widget)
        for edit in (self.name, self.event_name, self.event_code, self.folder):
            edit.textChanged.connect(self._sync)
        self.name.textChanged.connect(self._name_changed)
        self.folder.textEdited.connect(lambda *_: setattr(self, "_folder_edited", True))
        return page

    def _build_videos(self) -> QWidget:
        page = widgets.card()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(widgets.label("Drop videos here, or Browse", role="section"))
        layout.addWidget(widgets.label("Videos stay where they are; the project stores their paths only.", role="muted"))
        self.video_list = DropList()
        self.video_list.setAccessibleName("Videos to add")
        self.video_list.files_dropped.connect(self.add_videos)
        self.video_list.currentRowChanged.connect(self._sync)
        layout.addWidget(self.video_list, 1)
        row = QHBoxLayout()
        row.addWidget(widgets.button("Browse…", "primary", on_click=self._browse_videos))
        self.remove_video_btn = widgets.button("Remove selected", on_click=self._remove_video)
        row.addWidget(self.remove_video_btn)
        row.addStretch(1)
        layout.addLayout(row)
        link_row = QHBoxLayout()
        self.youtube = QLineEdit(); self.youtube.setPlaceholderText("Or paste a YouTube link")
        self.youtube.setAccessibleName("YouTube link")
        link_row.addWidget(self.youtube, 1)
        link_row.addWidget(widgets.button("Check link", on_click=self._check_link))
        layout.addLayout(link_row)
        self.youtube_status = widgets.label("", wrap=True)
        self.youtube_status.hide()
        layout.addWidget(self.youtube_status)
        return page

    def _build_models(self) -> QWidget:
        page = widgets.card()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(widgets.label("Choose the detector model for this project", role="section"))
        layout.addWidget(widgets.label("The choice is stored in the project, so changing the active model later "
                                       "does not rewrite this project's history.", role="muted", wrap=True))
        self.recommendation = widgets.label("", tone="info", wrap=True)
        layout.addWidget(self.recommendation)
        self.model_list = QListWidget()
        self.model_list.setAccessibleName("Model versions")
        self.model_list.setWordWrap(True)
        self.model_list.currentItemChanged.connect(self._sync)
        layout.addWidget(self.model_list, 1)
        self.no_models = widgets.EmptyState("No models yet — train one first",
                                            "Add an ONNX model file you already have, or open Models to train one.",
                                            "Open Models", self.open_models.emit, primary=False)
        layout.addWidget(self.no_models, 1)
        row = QHBoxLayout()
        row.addWidget(widgets.button("Add model file…", on_click=self._add_model_file))
        row.addStretch(1)
        layout.addLayout(row)
        return page

    def _build_confirm(self) -> QWidget:
        page = widgets.card()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(widgets.label("Confirm", role="section"))
        self.summary = widgets.label("", wrap=True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setAccessibleName("Project summary")
        layout.addWidget(self.summary)
        layout.addStretch(1)
        return page

    # ----- state -----

    def reset(self) -> None:
        """Start over with empty fields; nothing from a cancelled wizard is kept."""
        self.step = 0
        self.videos = []
        self._folder_edited = False
        for edit in (self.name, self.event_name, self.event_code, self.folder, self.youtube):
            edit.blockSignals(True); edit.clear(); edit.blockSignals(False)
        self.notes.clear()
        self.date.setDate(QDate.currentDate())
        self.youtube_status.hide()
        self.use_suggestion.hide()  # The fields were cleared with signals blocked.
        self.error.hide()
        self._refresh_videos()
        self.model_list.clear()  # Otherwise the refresh would restore the last pick.
        self._refresh_models()
        self._show_step()

    def spec(self) -> ProjectSpec:
        item = self.model_list.currentItem()
        model = None
        if item is not None:
            model = next((m for m in self.model_store.versions() if m.id == item.data(Qt.ItemDataRole.UserRole)), None)
            if model is None:  # Deleted after it was picked: keep a stub so the step says so.
                model = ModelVersion(id=item.data(Qt.ItemDataRole.UserRole), name="", path="")
        return ProjectSpec(
            name=self.name.text(), event_name=self.event_name.text(), event_code=self.event_code.text().strip() or None,
            event_date=self.date.date().toString("yyyy-MM-dd"), notes=self.notes.toPlainText(),
            folder=Path(self.folder.text().strip()) if self.folder.text().strip() else None,
            videos=list(self.videos), model=model)

    def problem(self, step: int | None = None) -> str | None:
        step = self.step if step is None else step
        spec = self.spec()
        return step_problem(step, spec, self.existing_names(), self.model_store.versions())

    def _show_step(self):
        self.pages.setCurrentIndex(self.step)
        for index, (number, text) in enumerate(self.step_labels):
            state = "done" if index < self.step else "current" if index == self.step else None
            theme.set_state(number, state)
            number.setText("✓" if index < self.step else str(index + 1))
        if self.step == 3:
            self._recheck_videos()
            self._refresh_summary()
        if self.step == 2:
            self._refresh_models()
        self._sync()

    def _sync(self, *_):
        problem = self.problem()
        last = self.step == len(STEPS) - 1
        self.back_btn.setEnabled(self.step > 0)
        self.next_btn.setVisible(not last)
        self.create_btn.setVisible(last)
        all_problems = [p for p in (self.problem(index) for index in range(3)) if p]
        widgets.disable_with_reason(self.next_btn, problem is None, problem or "")
        widgets.disable_with_reason(self.create_btn, last and not all_problems, all_problems[0] if all_problems else "")
        shown = problem if not last else (all_problems[0] if all_problems else None)
        self.reason.setText(shown or "")
        self.remove_video_btn.setEnabled(self.video_list.currentRow() >= 0)

    def next(self):
        if self.problem() is None and self.step < len(STEPS) - 1:
            self.step += 1
            self._show_step()

    def back(self):
        if self.step > 0:
            self.step -= 1
            self._show_step()

    def cancel(self):
        self.reset()
        self.cancelled.emit()

    def _recheck_videos(self):
        """Mark videos that were moved or deleted after they were added."""
        for video in self.videos:
            if not video.error and not video.path.is_file():
                video.error = "File moved or deleted"
        self._refresh_videos()

    def create(self):
        self._recheck_videos()
        problems = [p for p in (self.problem(index) for index in range(3)) if p]
        if problems:
            self._sync()
            return
        try:
            path = create_project(self.spec())
        except Exception as exc:
            self.error.setText(f"✕ Could not create the project: {exc}")
            self.error.show()
            return
        self.reset()
        self.created.emit(path)

    # ----- step 1 helpers -----

    def _name_changed(self, text):
        if not self._folder_edited:
            self.folder.blockSignals(True)
            safe = re.sub(r'[<>:"/\\|?*]+', "-", text.strip())
            self.folder.setText(str(self.default_folder(safe)) if safe else "")
            self.folder.blockSignals(False)
        existing = self.existing_names()
        if text.strip() and text.strip().casefold() in {name.casefold() for name in existing}:
            suggestion = unique_name(text, existing)
            self.use_suggestion.setText(f"Use \"{suggestion}\"")
            self.use_suggestion.setProperty("suggestion", suggestion)
            self.use_suggestion.show()
        else:
            self.use_suggestion.hide()
        self._sync()

    def _use_suggestion(self):
        self.name.setText(self.use_suggestion.property("suggestion") or self.name.text())

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose project folder")
        if folder:
            self._folder_edited = True
            self.folder.setText(folder)

    # ----- step 2 helpers -----

    def add_videos(self, paths):
        known = {video.path.resolve() for video in self.videos}
        for path in paths:
            path = Path(path)
            if path.resolve() in known:
                continue
            self.videos.append(check_video(path))
            known.add(path.resolve())
        self._refresh_videos()
        self._sync()

    def _browse_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add match videos", "", "Videos (*.mp4 *.mov *.mkv)")
        if files:
            self.add_videos([Path(file) for file in files])

    def _remove_video(self):
        row = self.video_list.currentRow()
        if 0 <= row < len(self.videos):
            del self.videos[row]  # Only removed from the list; the file is untouched.
            self._refresh_videos()
            self._sync()

    def _refresh_videos(self):
        self.video_list.clear()
        for video in self.videos:
            item = QListWidgetItem(video.describe())
            item.setToolTip(str(video.path))
            self.video_list.addItem(item)

    def _check_link(self):
        link = self.youtube.text().strip()
        self.youtube_status.show()
        if not YOUTUBE.match(link):
            theme.set_tone(self.youtube_status, "danger")
            self.youtube_status.setText("✕ That link did not load, check it and try again")
            return
        theme.set_tone(self.youtube_status, "pending")
        self.youtube_status.setText(f"⚠ {link}\nDownloading from YouTube is not available in this build, and nothing "
                                    "is downloaded without your OK. Save the video as an MP4, then add the file.")

    # ----- step 3 helpers -----

    def _refresh_models(self):
        current = self.model_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.model_list.currentItem() else None
        models = self.model_store.versions()
        recommended, reason = self.model_store.recommended()
        self.model_list.blockSignals(True)
        self.model_list.clear()
        for model in models:
            item = QListWidgetItem(model_row_text(model, recommended is not None and model.id == recommended.id))
            item.setData(Qt.ItemDataRole.UserRole, model.id)
            self.model_list.addItem(item)
            if model.id == current:
                self.model_list.setCurrentItem(item)
        self.model_list.blockSignals(False)
        if current and not any(model.id == current for model in models):
            stale = QListWidgetItem("✕ That model is gone, pick another")
            stale.setData(Qt.ItemDataRole.UserRole, current)
            stale.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.model_list.addItem(stale)
            self.model_list.setCurrentItem(stale)
        self.recommendation.setText(
            f"● Recommended: {recommended.name}. {reason} It is not selected for you." if recommended
            else f"No recommendation: {reason}")
        self.model_list.setVisible(bool(models))
        self.no_models.setVisible(not models)
        self.recommendation.setVisible(bool(models))

    def _add_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add a model file", "", "ONNX model (*.onnx);;All files (*)")
        if not path:
            return
        try:
            model = self.model_store.add_file(Path(path))
        except (OSError, ValueError) as exc:
            self.error.setText(f"✕ Could not add the model: {exc}")
            self.error.show()
            return
        self._refresh_models()
        for row in range(self.model_list.count()):
            if self.model_list.item(row).data(Qt.ItemDataRole.UserRole) == model.id:
                self.model_list.setCurrentRow(row)
        self._sync()

    # ----- step 4 -----

    def _refresh_summary(self):
        spec = self.spec()
        usable = [video for video in spec.videos if video.usable]
        total = sum(video.duration_s or 0 for video in usable)
        skipped = len(spec.videos) - len(usable)
        # Names and paths are typed by the user: escape them so "&" or "<" show as written.
        event = html.escape(spec.event_name.strip())
        if spec.event_code:
            event += f" ({html.escape(normalise_event_code(spec.event_code))})"
        lines = [
            f"<b>Name</b>: {html.escape(spec.name.strip())}",
            f"<b>Event</b>: {event}",
            f"<b>Date</b>: {html.escape(spec.event_date)}",
            f"<b>Season</b>: {html.escape(SEASON)}",
            f"<b>Videos</b>: {len(usable)}, total {int(total // 60)} min {int(total % 60)} s"
            + (f" ({skipped} unreadable or missing file(s) will be skipped)" if skipped else ""),
            f"<b>Model</b>: {html.escape(spec.model.name) if spec.model else '—'}",
            f"<b>Folder</b>: {html.escape(str(spec.folder))}",
        ]
        self.summary.setText("<br>".join(lines))
