"""Page 5 — Livestream splitter: turn one long stream into per-match videos, confirmed by a person."""
from __future__ import annotations

import json
import sqlite3
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication, QRectF, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QSizePolicy, QVBoxLayout, QWidget,
)

from control_vision.app import theme, widgets
from control_vision.app.media import format_clock, parse_clock

SHORT_MATCH_S = 60.0        # under a minute is probably a false split
MIN_LENGTH_S = 1.0
DRAFT_FILE = "livestream_splits.json"


@dataclass
class Segment:
    start: float
    end: float
    match: str = ""          # typed by a person, or read from the video; never guessed
    confirmed: bool = False

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def short(self) -> bool:
        return self.length < SHORT_MATCH_S


def add_segment(segments: list[Segment], start: float, end: float, duration: float) -> list[Segment]:
    """A new list with [start, end] added in time order; overlaps are refused."""
    start, end = max(0.0, start), min(duration, end)
    if end - start < MIN_LENGTH_S:
        raise ValueError("The end of a match must be at least a second after its start.")
    for other in segments:
        if start < other.end and other.start < end:
            raise ValueError(f"That overlaps the match at {format_clock(other.start)}–{format_clock(other.end)}.")
    return sorted([*segments, Segment(start, end)], key=lambda segment: segment.start)


def adjust_segment(segments: list[Segment], index: int, start: float | None = None, end: float | None = None,
                   duration: float = float("inf")) -> None:
    """Move one boundary, kept between the neighbours and at least a second long. Unconfirms the match."""
    segment = segments[index]
    low = segments[index - 1].end if index > 0 else 0.0
    high = segments[index + 1].start if index + 1 < len(segments) else duration
    if start is not None:
        segment.start = min(max(low, start), segment.end - MIN_LENGTH_S)
    if end is not None:
        segment.end = max(min(high, end), segment.start + MIN_LENGTH_S)
    segment.confirmed = False


def merge_segments(segments: list[Segment], index: int) -> list[Segment]:
    """Join a match with the next one (a camera cut split it); the merged match needs confirming again."""
    if not 0 <= index < len(segments) - 1:
        raise ValueError("Select a match that has another match after it.")
    first, second = segments[index], segments[index + 1]
    numbers = [number for number in (first.match, second.match) if number]
    merged = Segment(first.start, second.end, numbers[0] if len(set(numbers)) == 1 else "", False)
    return [*segments[:index], merged, *segments[index + 2:]]


def clip_name(source: Path, segment: Segment) -> str:
    """Named by where the match starts in the stream, so it never depends on which splits are left."""
    match = re.sub(r"[^A-Za-z0-9_-]+", "", segment.match)
    start = format_clock(segment.start).replace(":", "-")
    return f"{source.stem}-at{start}" + (f"-{match}" if match else "") + ".mp4"


def load_draft(project_dir: Path, source: Path) -> list[Segment]:
    try:
        data = json.loads((project_dir / DRAFT_FILE).read_text(encoding="utf-8"))
        entries = data.get(str(Path(source).resolve()), []) if isinstance(data, dict) else []
        segments = [Segment(float(e["start"]), float(e["end"]), str(e.get("match", "")), bool(e.get("confirmed")))
                    for e in entries if isinstance(e, dict)]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return sorted((s for s in segments if s.end - s.start >= MIN_LENGTH_S), key=lambda s: s.start)


def save_draft(project_dir: Path, source: Path, segments: list[Segment]) -> None:
    path = project_dir / DRAFT_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    data[str(Path(source).resolve())] = [asdict(segment) for segment in segments]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


class Timeline(QWidget):
    """The whole stream as a bar: matches as blocks, a playhead, draggable match edges."""

    seek = Signal(float)
    selected = Signal(int)
    boundary_moved = Signal(int, str, float)   # index, "start" or "end", seconds
    boundary_released = Signal()

    GRAB_PX = 6

    def __init__(self):
        super().__init__()
        self.duration = 0.0
        self.position = 0.0
        self.segments: list[Segment] = []
        self.current = -1
        self.mark_in: float | None = None
        self._drag: tuple[int, str] | None = None
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Stream timeline")

    def _x(self, seconds: float) -> float:
        return 8 + (self.width() - 16) * (seconds / self.duration if self.duration else 0)

    def _seconds(self, x: float) -> float:
        return max(0.0, min(self.duration, (x - 8) / max(1, self.width() - 16) * self.duration))

    def _edge_at(self, x: float) -> tuple[int, str] | None:
        for index, segment in enumerate(self.segments):
            for side, seconds in (("start", segment.start), ("end", segment.end)):
                if abs(self._x(seconds) - x) <= self.GRAB_PX:
                    return index, side
        return None

    def paintEvent(self, event):
        tokens = _tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = QRectF(8, 18, self.width() - 16, 22)
        painter.fillRect(bar, QColor(tokens.surface_alt))
        painter.setPen(QPen(QColor(tokens.border)))
        painter.drawRect(bar)
        for index, segment in enumerate(self.segments):
            rect = QRectF(self._x(segment.start), bar.top(), max(2.0, self._x(segment.end) - self._x(segment.start)),
                          bar.height())
            colour = tokens.success_fg if segment.confirmed and not segment.short else tokens.pending_fg
            painter.fillRect(rect, QColor(colour))
            if index == self.current:
                painter.setPen(QPen(QColor(tokens.accent), 2))
                painter.drawRect(rect.adjusted(-1, -3, 1, 3))
        if self.mark_in is not None:
            painter.setPen(QPen(QColor(tokens.accent), 2, Qt.PenStyle.DashLine))
            x = self._x(self.mark_in)
            painter.drawLine(int(x), 10, int(x), 46)
        painter.setPen(QPen(QColor(tokens.text), 2))
        x = self._x(self.position)
        painter.drawLine(int(x), 6, int(x), 50)
        painter.setPen(QColor(tokens.text_muted))
        painter.drawText(8, 12, "00:00")
        end = format_clock(self.duration)
        painter.drawText(int(self.width() - 8 - painter.fontMetrics().horizontalAdvance(end)), 12, end)
        painter.end()

    def mousePressEvent(self, event):
        if not self.duration:
            return
        x = event.position().x()
        edge = self._edge_at(x)
        if edge is not None:
            self._drag = edge
            self.selected.emit(edge[0])
            return
        seconds = self._seconds(x)
        for index, segment in enumerate(self.segments):
            if segment.start <= seconds <= segment.end:
                self.selected.emit(index)
                break
        self.seek.emit(seconds)

    def mouseMoveEvent(self, event):
        x = event.position().x()
        if self._drag is not None:
            self.boundary_moved.emit(self._drag[0], self._drag[1], self._seconds(x))
            return
        self.setCursor(Qt.CursorShape.SizeHorCursor if self._edge_at(x) else Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self._drag = None
            self.boundary_released.emit()


def _tokens():
    from PySide6.QtWidgets import QApplication
    return theme.resolve_tokens(QApplication.instance().property("control_vision_theme") or theme.DEFAULT_MODE)


class ClipWorker(QThread):
    """Cuts confirmed matches into new files off the UI thread."""

    progress = Signal(int, int)
    finished_clips = Signal(list)    # [(segment, path, metadata)]
    failed = Signal(str)

    def __init__(self, source: Path, jobs: list[tuple[Segment, Path]], ffmpeg: str | None):
        super().__init__()
        self.source, self.jobs, self.ffmpeg = source, jobs, ffmpeg
        self.cancelled = False

    def run(self):
        from control_vision.app.media import cut_clip
        from control_vision.pipeline.ingest import probe_video
        done = []
        try:
            for number, (segment, output) in enumerate(self.jobs, start=1):
                if self.cancelled:
                    break
                cut_clip(self.source, segment.start, segment.end, output, self.ffmpeg)
                try:
                    metadata = probe_video(output)
                except Exception:
                    output.unlink(missing_ok=True)  # Otherwise it would block a retry as "already exists".
                    raise
                done.append((segment, output, metadata))
                self.progress.emit(number, len(self.jobs))
        except Exception as exc:
            self.finished_clips.emit(done)
            self.failed.emit(str(exc))
            return
        self.finished_clips.emit(done)


class SplitterPage(QWidget):
    clips_created = Signal(list)     # ids of the new videos
    open_calibrate = Signal()
    new_project = Signal()

    def __init__(self, ffmpeg_path=lambda: None):
        super().__init__()
        self.setObjectName("Page")
        self.ffmpeg_path = ffmpeg_path
        self.db = None
        self.project_id = None
        self.project_dir: Path | None = None
        self.source: Path | None = None
        self.capture = None
        self.fps = 30.0
        self.duration = 0.0
        self.position = 0.0
        self.segments: list[Segment] = []
        self.worker: ClipWorker | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        self.no_project = widgets.EmptyState("Open a project to split a stream",
                                             "Split matches become new videos in the project.",
                                             "+ New project", self.new_project.emit)
        outer.addWidget(self.no_project, 1)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        outer.addWidget(self.body, 1)

        top = QHBoxLayout()
        self.source_label = widgets.label("No stream open", role="section")
        top.addWidget(self.source_label, 1)
        self.detect_btn = widgets.button("Detect matches")
        widgets.disable_with_reason(self.detect_btn, False,
                                    "Automatic match detection is not available in this build — split by hand")
        top.addWidget(self.detect_btn)
        top.addWidget(widgets.button("Open stream…", "primary", on_click=self.browse))
        body.addLayout(top)

        self.player = QLabel()
        self.player.setObjectName("Thumbnail")
        self.player.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.player.setMinimumSize(480, 270)
        self.player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.player.setAccessibleName("Stream player")
        self.player.setText("Open a long stream file to split it into matches")
        body.addWidget(self.player, 3)

        transport = QHBoxLayout()
        self.play_btn = widgets.button("Play", on_click=self.toggle_play)
        transport.addWidget(self.play_btn)
        for text, step in (("−1 min", -60), ("−10 s", -10), ("−1 s", -1), ("+1 s", 1), ("+10 s", 10), ("+1 min", 60)):
            transport.addWidget(widgets.button(text, on_click=lambda s=step: self.seek(self.position + s)))
        self.time_label = widgets.label("00:00 / 00:00", role="muted")
        transport.addWidget(self.time_label)
        transport.addStretch(1)
        self.mark_in_btn = widgets.button("Mark In", on_click=self.mark_in)
        self.mark_out_btn = widgets.button("Mark Out", on_click=self.mark_out)
        transport.addWidget(self.mark_in_btn)
        transport.addWidget(self.mark_out_btn)
        body.addLayout(transport)

        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek)
        self.timeline.selected.connect(self.select)
        self.timeline.boundary_moved.connect(self._drag_boundary)
        self.timeline.boundary_released.connect(self._changed)
        body.addWidget(self.timeline)

        manual = QHBoxLayout()
        manual.addWidget(widgets.label("Manual split", role="section"))
        self.start_edit = QLineEdit(); self.start_edit.setPlaceholderText("Start, e.g. 1:23:45")
        self.end_edit = QLineEdit(); self.end_edit.setPlaceholderText("End, e.g. 1:26:15")
        for edit, name in ((self.start_edit, "Match start time"), (self.end_edit, "Match end time")):
            edit.setAccessibleName(name); edit.setFixedWidth(150)
            manual.addWidget(edit)
        manual.addWidget(widgets.button("Add match", on_click=self.add_typed))
        self.message = widgets.label("", wrap=True)
        manual.addWidget(self.message, 1)
        body.addLayout(manual)

        lower = QHBoxLayout()
        self.cards = QListWidget()
        self.cards.setAccessibleName("Matches in this stream")
        self.cards.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cards.currentRowChanged.connect(self._card_selected)
        lower.addWidget(self.cards, 2)
        side = QVBoxLayout()
        self.match_edit = QLineEdit(); self.match_edit.setPlaceholderText("Match number (blank = Unknown match)")
        self.match_edit.setAccessibleName("Match number")
        self.match_edit.editingFinished.connect(self._set_match_number)
        side.addWidget(self.match_edit)
        self.confirm_btn = widgets.button("Confirm split", "success", on_click=self.confirm_selected)
        self.jump_end_btn = widgets.button("Jump to end", on_click=lambda: self._jump("end"))
        self.merge_btn = widgets.button("Merge with next", on_click=self.merge_selected)
        self.delete_btn = widgets.button("Delete split", "danger-outline", on_click=self.delete_selected)
        for control in (self.confirm_btn, self.jump_end_btn, self.merge_btn, self.delete_btn):
            side.addWidget(control)
        side.addStretch(1)
        lower.addLayout(side, 1)
        body.addLayout(lower, 2)

        self.empty = widgets.label("No matches detected yet — try manual split: Mark In and Mark Out while playing, "
                                   "or type a start and end.", role="muted", wrap=True)
        body.addWidget(self.empty)
        footer = QHBoxLayout()
        self.summary = widgets.label("", role="muted")
        footer.addWidget(self.summary, 1)
        self.progress = QProgressBar(); self.progress.setFixedWidth(200); self.progress.hide()
        footer.addWidget(self.progress)
        self.cancel_btn = widgets.button("Cancel", on_click=self.cancel_clips)
        self.cancel_btn.hide()
        footer.addWidget(self.cancel_btn)
        self.calibrate_btn = widgets.button("Calibrate ›", on_click=self.open_calibrate.emit)
        self.calibrate_btn.hide()
        footer.addWidget(self.calibrate_btn)
        self.create_btn = widgets.button("Create match videos", "primary", on_click=self.create_clips)
        footer.addWidget(self.create_btn)
        body.addLayout(footer)
        self.set_project(None, None, None)

    # ----- project and source -----

    def set_project(self, db, project_id, project_dir) -> None:
        if (db, project_id) != (self.db, self.project_id):
            self.close_source()
        self.db, self.project_id = db, project_id
        self.project_dir = Path(project_dir) if project_dir else None
        has = db is not None and project_id is not None
        self.no_project.setVisible(not has)
        self.body.setVisible(has)
        self._render()

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a livestream file", "", "Videos (*.mp4 *.mov *.mkv)")
        if path:
            self.open_source(Path(path))

    def open_source(self, path: Path) -> bool:
        self.close_source()
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if not capture.isOpened() or fps <= 0 or frames <= 0:
            capture.release()
            self._say(f"✕ Cannot read {Path(path).name}.", "danger")
            return False
        self.capture, self.source, self.fps, self.duration = capture, Path(path), fps, frames / fps
        self.segments = load_draft(self.project_dir, self.source) if self.project_dir else []
        self.source_label.setText(f"{self.source.name} · {format_clock(self.duration)}")
        self.seek(0)
        self._render()
        if self.segments:
            self._say(f"Restored {len(self.segments)} match(es) from your saved draft.", "info")
        return True

    def close_source(self) -> None:
        self.timer.stop()
        self.play_btn.setText("Play")
        if self.capture is not None:
            self.capture.release()
        self.capture, self.source, self.duration, self.position, self.segments = None, None, 0.0, 0.0, []

    # ----- player -----

    def seek(self, seconds: float) -> None:
        if self.capture is None:
            return
        self.position = max(0.0, min(self.duration, float(seconds)))
        self.capture.set(cv2.CAP_PROP_POS_MSEC, self.position * 1000)
        ok, frame = self.capture.read()
        if ok:
            self._show(frame)
        self._update_clock()

    def toggle_play(self) -> None:
        if self.capture is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start()
            self.play_btn.setText("Pause")

    def _tick(self) -> None:
        # Read forward instead of seeking every tick, so playback stays smooth on long files.
        step = max(1, round(self.fps * self.timer.interval() / 1000))
        frame = None
        for _ in range(step):
            ok, frame = self.capture.read()
            if not ok:
                self.toggle_play()
                return
        self.position = min(self.duration, self.position + step / self.fps)
        self._show(frame)
        self._update_clock()

    def _show(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.player.setPixmap(QPixmap.fromImage(image).scaled(
            self.player.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _update_clock(self) -> None:
        self.time_label.setText(f"{format_clock(self.position)} / {format_clock(self.duration)}")
        self.timeline.duration, self.timeline.position = self.duration, self.position
        self.timeline.update()

    # ----- splits -----

    def mark_in(self) -> None:
        self.timeline.mark_in = self.position
        self._say(f"Mark In at {format_clock(self.position)}. Play to the end of the match, then Mark Out.", "info")
        self.timeline.update()

    def mark_out(self) -> None:
        if self.timeline.mark_in is None:
            self._say("Mark In first, at the start of the match.", "pending")
            return
        self._add(self.timeline.mark_in, self.position)

    def add_typed(self) -> None:
        try:
            start, end = parse_clock(self.start_edit.text()), parse_clock(self.end_edit.text())
        except ValueError as exc:
            self._say(f"✕ {exc}", "danger")
            return
        if self._add(start, end):
            self.start_edit.clear(); self.end_edit.clear()

    def _add(self, start: float, end: float) -> bool:
        if self.source is None:
            self._say("Open a stream first.", "pending")
            return False
        try:
            self.segments = add_segment(self.segments, start, end, self.duration)
        except ValueError as exc:
            self._say(f"✕ {exc}", "danger")
            return False
        self.timeline.mark_in = None
        index = next(i for i, s in enumerate(self.segments) if s.start == max(0.0, start))
        self._changed()
        self.select(index)
        self._say(f"Added a match at {format_clock(start)}–{format_clock(end)}. Check it, then Confirm split.", "info")
        return True

    def _drag_boundary(self, index: int, side: str, seconds: float) -> None:
        adjust_segment(self.segments, index, **{side: seconds}, duration=self.duration)
        self.timeline.segments = self.segments
        self.timeline.update()
        self._render_cards()
        self.seek(getattr(self.segments[index], side))

    def select(self, index: int) -> None:
        if 0 <= index < len(self.segments) and self.cards.currentRow() != index:
            self.cards.setCurrentRow(index)       # _card_selected follows.
        else:
            # Same row after a delete or merge: it is another split now, so show its number and time.
            self._card_selected(index)

    def _card_selected(self, index: int) -> None:
        self.timeline.current = index
        self.timeline.update()
        if 0 <= index < len(self.segments):
            segment = self.segments[index]
            self.match_edit.setText(segment.match)
            self.seek(segment.start)
        else:
            self.match_edit.clear()
        self._sync_actions()

    def _jump(self, side: str) -> None:
        index = self.cards.currentRow()
        if 0 <= index < len(self.segments):
            self.seek(getattr(self.segments[index], side))

    def _set_match_number(self) -> None:
        index = self.cards.currentRow()
        if 0 <= index < len(self.segments) and self.segments[index].match != self.match_edit.text().strip():
            self.segments[index].match = self.match_edit.text().strip()
            self._changed()

    def confirm_selected(self) -> None:
        index = self.cards.currentRow()
        if 0 <= index < len(self.segments):
            self.segments[index].confirmed = True
            self._changed()
            following = next((i for i, s in enumerate(self.segments) if not s.confirmed), None)
            if following is not None:
                self.select(following)

    def merge_selected(self) -> None:
        index = self.cards.currentRow()
        try:
            self.segments = merge_segments(self.segments, index)
        except ValueError as exc:
            self._say(f"✕ {exc}", "danger")
            return
        self._changed()
        self.select(index)

    def delete_selected(self) -> None:
        index = self.cards.currentRow()
        if 0 <= index < len(self.segments):
            del self.segments[index]
            self._changed()
            self.select(min(index, len(self.segments) - 1))

    def _changed(self) -> None:
        if self.project_dir is not None and self.source is not None:
            try:
                save_draft(self.project_dir, self.source, self.segments)
            except OSError as exc:
                self._say(f"⚠ The draft could not be saved: {exc}", "pending")
        self._render()

    # ----- rendering -----

    def _say(self, text: str, tone: str) -> None:
        theme.set_tone(self.message, tone)
        self.message.setText(text)

    def _render(self) -> None:
        self.timeline.segments = self.segments
        self.timeline.update()
        self._render_cards()
        has_source = self.source is not None
        for control in (self.play_btn, self.mark_in_btn, self.mark_out_btn, self.start_edit, self.end_edit):
            control.setEnabled(has_source)
        self.empty.setVisible(has_source and not self.segments)
        confirmed = [s for s in self.segments if s.confirmed]
        check = [s for s in self.segments if not s.confirmed or s.short]
        if not self.segments:
            self.summary.setText("")
        elif not check:
            self.summary.setText(f"✓ All matches confirmed ({len(confirmed)})")
        else:
            self.summary.setText(f"{len(confirmed)} of {len(self.segments)} confirmed · {len(check)} to check")
        busy = self.worker is not None
        widgets.disable_with_reason(self.create_btn, bool(confirmed) and not busy,
                                    "Creating videos…" if busy else "Confirm at least one match first")
        self.create_btn.setText(f"Create {len(confirmed)} match video" + ("" if len(confirmed) == 1 else "s")
                                if confirmed else "Create match videos")
        self._sync_actions()

    def _render_cards(self) -> None:
        current = self.cards.currentRow()
        self.cards.blockSignals(True)
        self.cards.clear()
        for index, segment in enumerate(self.segments):
            if segment.short:
                state = "⚑ Very short — check this"
            elif segment.confirmed:
                state = "✓ Confirmed"
            else:
                state = "⚑ Check this one"
            name = f"Match {segment.match}" if segment.match else "Unknown match"
            item = QListWidgetItem(f"{index + 1}. {name} · {format_clock(segment.start)}–{format_clock(segment.end)} "
                                   f"({format_clock(segment.length)}) · {state}")
            item.setForeground(QColor(_tokens().success_fg if segment.confirmed and not segment.short
                                      else _tokens().pending_fg))
            self.cards.addItem(item)
        if 0 <= current < self.cards.count():
            self.cards.setCurrentRow(current)
        self.cards.blockSignals(False)

    def _sync_actions(self) -> None:
        index = self.cards.currentRow()
        selected = 0 <= index < len(self.segments)
        for control in (self.confirm_btn, self.jump_end_btn, self.delete_btn, self.match_edit):
            widgets.disable_with_reason(control, selected, "Select a match in the list")
        widgets.disable_with_reason(self.merge_btn, selected and index < len(self.segments) - 1,
                                    "Select a match that has another match after it")

    # ----- creating the match videos -----

    def create_clips(self) -> None:
        if self.source is None or self.db is None or self.project_dir is None or self.worker is not None:
            return
        out_dir = self.project_dir / "matches"
        jobs = []
        for segment in self.segments:
            if segment.confirmed:
                output = out_dir / clip_name(self.source, segment)
                if output.exists():
                    self._say(f"✕ {output.name} already exists in {out_dir}. Delete or move it first.", "danger")
                    return
                jobs.append((segment, output))
        self._clip_target = (self.db, self.project_id)  # Even if the project changes while cutting.
        self.worker = ClipWorker(self.source, jobs, self.ffmpeg_path())
        self.worker.progress.connect(lambda done, total: self.progress.setValue(round(100 * done / total)))
        self.worker.finished_clips.connect(self._clips_done)
        self.worker.failed.connect(lambda message: self._say(f"✕ Could not create every video: {message}", "danger"))
        self.progress.setValue(0); self.progress.show(); self.cancel_btn.show()
        self._say(f"Creating {len(jobs)} match video(s) in {out_dir}…", "info")
        self._render()
        self.worker.start()

    def stop_clips(self) -> None:
        """Stop cutting and add what was cut, before the window closes the database."""
        if self.worker is None:
            return
        self.worker.cancelled = True
        self.worker.wait()
        QCoreApplication.sendPostedEvents()  # Delivers finished_clips, which adds the rows.
        self.worker = None

    def cancel_clips(self) -> None:
        if self.worker is not None:
            self.worker.cancelled = True
            self._say("Stopping after the current video…", "pending")

    def _clips_done(self, done: list) -> None:
        db, project_id = getattr(self, "_clip_target", (self.db, self.project_id))
        same_project = (db, project_id) == (self.db, self.project_id)
        ids, error = [], None
        try:
            details = db.project_details(project_id) or {}
            for segment, path, metadata in done:
                video_id = db.add_video(project_id, path, metadata, details.get("event_code"), segment.match or None)
                ids.append(video_id)
                if same_project:  # Cut and added: it is no longer a split to cut.
                    self.segments = [other for other in self.segments if other is not segment]
                try:
                    db.set_video_details(video_id, file_size=Path(path).stat().st_size)
                except OSError:
                    pass  # The size is only shown in tables.
        except (sqlite3.Error, OSError, ValueError) as exc:
            error = exc
        finally:
            if self.worker is not None:
                self.worker.wait()
            self.worker = None
            self.progress.hide(); self.cancel_btn.hide()
        if not same_project:
            # The project was switched while cutting: the videos stay with the project they were cut for.
            self._render()
            return
        self._changed()
        if ids:
            self._say(f"✓ Added {len(ids)} match video(s) to the project. Clips start at the nearest keyframe, "
                      "so they can begin a moment early.", "success")
            self.calibrate_btn.show()
            self.clips_created.emit(ids)
        if error is not None:
            self._say(f"✕ Could not add every match video to the project: {error}", "danger")
