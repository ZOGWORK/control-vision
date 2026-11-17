"""Page 7 — Identity: which robot is which team and alliance, before anything is counted."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIntValidator, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox, QSlider,
    QVBoxLayout, QWidget,
)

from control_vision.app import theme, widgets
from control_vision.app.frameview import FrameView, frame_pixmap, tokens
from control_vision.app.media import read_frame

ROBOTS_PER_MATCH = 4
# The tracker matches each click to a detected robot only in the first seconds of
# the video (seed_assignment_deadline_s in pipeline/tracking.py), so the frame to
# click on has to come from that window.
SEED_WINDOW_S = 3.0
POSITIONS = (("Not set", None), ("Left", "left"), ("Right", "right"))


@dataclass
class RobotEntry:
    id: int | None = None
    team: str = ""
    alliance: str | None = None      # chosen by a person; never defaulted
    start_position: str | None = None
    absent: bool = False
    seed: tuple[int, int] | None = None

    @property
    def empty(self) -> bool:
        return not self.team and self.alliance is None and self.seed is None and not self.absent

    @property
    def complete(self) -> bool:
        return bool(self.team) and self.alliance is not None and (self.absent or self.seed is not None)


def identity_state(entries: list[RobotEntry]) -> tuple[str, str, bool]:
    """(message, tone, savable). Only a duplicate team or a half-filled card blocks saving."""
    teams = [entry.team for entry in entries if entry.team]
    duplicates = sorted({team for team in teams if teams.count(team) > 1})
    if duplicates:
        return f"✕ Team {duplicates[0]} is already assigned. Each team can be assigned once.", "danger", False
    for number, entry in enumerate(entries, start=1):
        if entry.empty or entry.complete:
            continue
        if not entry.team:
            return f"Robot {number}: type its team number.", "pending", False
        if entry.alliance is None:
            return f"Robot {number}: choose Red or Blue.", "pending", False
        return f"Robot {number}: click it on the frame, or mark it not visible yet.", "pending", False
    filled = [entry for entry in entries if entry.complete]
    if not filled:
        return "Click a robot on the frame, then type its team and pick its alliance.", "info", False
    if not any(not entry.absent for entry in filled):
        return "At least one robot has to be visible to process the match.", "pending", True
    notes = []
    red = sum(entry.alliance == "red" for entry in filled)
    blue = sum(entry.alliance == "blue" for entry in filled)
    if red > 2 or blue > 2:
        notes.append(f"{red} Red, {blue} Blue — check this, a match is normally 2v2.")
    absent = sum(entry.absent for entry in filled)
    if len(filled) < ROBOTS_PER_MATCH:
        missing = ROBOTS_PER_MATCH - len(filled)
        notes.insert(0, f"{len(filled)} of {ROBOTS_PER_MATCH} robots set — fill the other {missing} or mark them not "
                        "visible, so the match is scored with all four accounted for.")
    elif absent:
        notes.insert(0, f"{absent} robot{'s' if absent > 1 else ''} not visible — you can still process; "
                        f"{'they' if absent > 1 else 'it'} will be marked as absent.")
    if notes:
        return " ".join(notes), "pending", True
    return "✓ Ready to process.", "success", True


def crop_around(frame: np.ndarray, x: int, y: int, half: int = 70) -> np.ndarray:
    height, width = frame.shape[:2]
    left, top = max(0, x - half), max(0, y - half)
    return frame[top:min(height, y + half), left:min(width, x + half)]


class IdentityCanvas(FrameView):
    """The frame with a numbered marker on each clicked robot, in its alliance colour."""

    def __init__(self):
        super().__init__("Choose a video to assign robots")
        self.entries: list[RobotEntry] = []
        self.current = 0
        self.loupe_enabled = True

    def paint_overlay(self, painter: QPainter) -> None:
        colours = tokens()
        font = QFont(painter.font()); font.setBold(True); painter.setFont(font)
        for index, entry in enumerate(self.entries):
            if entry.seed is None or entry.absent:
                continue
            point = self.to_widget(*entry.seed)
            colour = {"red": colours.red, "blue": colours.blue}.get(entry.alliance, colours.text)
            if index == self.current:
                painter.setPen(QPen(QColor(colours.accent), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(point, 22, 22)
            painter.setPen(QPen(QColor(colour), 3))
            painter.setBrush(QColor(0, 0, 0, 140))
            painter.drawEllipse(point, 14, 14)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(QRectF(point.x() - 14, point.y() - 14, 28, 28), Qt.AlignmentFlag.AlignCenter, str(index + 1))
            label = f"{entry.team or '?'}"
            painter.drawText(QRectF(point.x() + 18, point.y() - 10, 120, 20), Qt.AlignmentFlag.AlignLeft, label)


class RobotCard(QFrame):
    changed = Signal()
    chosen = Signal(int)
    swap = Signal(int, int)

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.setProperty("card", True)
        self.setProperty("interactive", True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        head = QHBoxLayout()
        self.title = widgets.label(f"Robot {index + 1}", role="section")
        head.addWidget(self.title, 1)
        self.swap_btn = widgets.button("Swap…", on_click=self._swap_menu)
        self.swap_btn.setToolTip("Swap team, alliance and position with another robot")
        head.addWidget(self.swap_btn)
        layout.addLayout(head)
        self.thumb = QLabel("Click this robot on the frame")
        self.thumb.setObjectName("Thumbnail")
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setWordWrap(True)
        self.thumb.setFixedHeight(110)
        layout.addWidget(self.thumb)
        self.team = QLineEdit(); self.team.setPlaceholderText("Team number")
        self.team.setValidator(QIntValidator(1, 99999, self))
        self.team.setAccessibleName(f"Robot {index + 1} team number")
        self.team.textEdited.connect(lambda *_: self.changed.emit())
        team_label = QLabel("&Team"); team_label.setBuddy(self.team)
        team_row = QHBoxLayout(); team_row.addWidget(team_label); team_row.addWidget(self.team, 1)
        layout.addLayout(team_row)
        alliance_row = QHBoxLayout()
        self.red = widgets.button("Red"); self.blue = widgets.button("Blue")
        self.alliances = QButtonGroup(self)
        for button, alliance in ((self.red, "red"), (self.blue, "blue")):
            button.setCheckable(True)
            button.setProperty("variant", "alliance")
            button.setProperty("alliance", alliance)
            button.setAccessibleName(f"Robot {index + 1} {alliance} alliance")
            self.alliances.addButton(button)
            alliance_row.addWidget(button, 1)
        self.alliances.buttonClicked.connect(lambda *_: self.changed.emit())
        layout.addLayout(alliance_row)
        self.position = QComboBox()
        for text, value in POSITIONS:
            self.position.addItem(text, value)
        self.position.setAccessibleName(f"Robot {index + 1} starting position")
        self.position.currentIndexChanged.connect(lambda *_: self.changed.emit())
        position_label = QLabel("&Start"); position_label.setBuddy(self.position)
        position_row = QHBoxLayout(); position_row.addWidget(position_label); position_row.addWidget(self.position, 1)
        layout.addLayout(position_row)
        self.absent = QCheckBox("Not visible yet")
        self.absent.setToolTip("The robot is missing or hidden in this frame. It is saved as absent.")
        self.absent.toggled.connect(lambda *_: self.changed.emit())
        layout.addWidget(self.absent)
        self.siblings: list[RobotCard] = []

    def mousePressEvent(self, event):
        self.chosen.emit(self.index)
        super().mousePressEvent(event)

    def focusInEvent(self, event):
        self.chosen.emit(self.index)
        super().focusInEvent(event)

    def _swap_menu(self) -> None:
        menu = QMenu(self)
        for other in self.siblings:
            if other is not self:
                menu.addAction(f"Swap with robot {other.index + 1}", lambda o=other.index: self.swap.emit(self.index, o))
        menu.exec(self.swap_btn.mapToGlobal(self.swap_btn.rect().bottomLeft()))

    def read(self, entry: RobotEntry) -> None:
        entry.team = self.team.text().strip()
        checked = self.alliances.checkedButton()
        entry.alliance = checked.property("alliance") if checked else None
        entry.start_position = self.position.currentData()
        entry.absent = self.absent.isChecked()

    def show_entry(self, entry: RobotEntry, frame: np.ndarray | None, current: bool) -> None:
        for control in (self.team, self.position, self.absent):
            control.blockSignals(True)
        self.team.setText(entry.team)
        self.alliances.setExclusive(False)
        self.red.setChecked(entry.alliance == "red"); self.blue.setChecked(entry.alliance == "blue")
        self.alliances.setExclusive(True)
        self.position.setCurrentIndex(max(0, self.position.findData(entry.start_position)))
        self.absent.setChecked(entry.absent)
        for control in (self.team, self.position, self.absent):
            control.blockSignals(False)
        alliance = entry.alliance if not entry.absent else None
        self.setProperty("alliance", alliance or "")
        self.setProperty("selected", current)
        theme.set_state(self, "absent" if entry.absent else None)
        self.style().unpolish(self); self.style().polish(self)
        self.title.setText(f"Robot {self.index + 1}" + (" · Red" if alliance == "red" else " · Blue" if alliance == "blue"
                                                         else " · Not visible yet" if entry.absent else ""))
        if entry.absent:
            self.thumb.clear(); self.thumb.setText("Not visible yet — saved as absent")
        elif entry.seed is not None and frame is not None:
            crop = crop_around(frame, *entry.seed)
            self.thumb.setPixmap(frame_pixmap(crop).scaled(self.thumb.width(), self.thumb.height(),
                                                           Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation))
        else:
            self.thumb.clear()
            self.thumb.setText("Select this card, then click the robot on the frame" if current
                               else "Click this robot on the frame")


class IdentityPage(QWidget):
    saved = Signal(int)
    continue_processing = Signal(int)
    new_project = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        self.db = None
        self.project_id = None
        self.video: dict | None = None
        self.entries = [RobotEntry() for _ in range(ROBOTS_PER_MATCH)]
        self.saved_entries: list[RobotEntry] = []
        self.dirty = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        self.no_project = widgets.EmptyState("Open a project to assign robots", "Robot identities are saved per match.",
                                             "+ New project", self.new_project.emit)
        outer.addWidget(self.no_project, 1)
        self.body = QWidget()
        body = QVBoxLayout(self.body); body.setContentsMargins(0, 0, 0, 0); body.setSpacing(10)
        outer.addWidget(self.body, 1)

        top = QHBoxLayout()
        video_label = QLabel("&Match"); top.addWidget(video_label)
        self.video_combo = QComboBox(); self.video_combo.setMinimumWidth(320)
        self.video_combo.setAccessibleName("Match video")
        video_label.setBuddy(self.video_combo)
        self.video_combo.activated.connect(self._video_chosen)
        top.addWidget(self.video_combo)
        top.addWidget(widgets.label(f"Use a clearer frame (first {SEED_WINDOW_S:.0f} s)", role="muted"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, int(SEED_WINDOW_S * 10))
        self.frame_slider.setAccessibleName("Frame to assign robots on")
        self.frame_slider.setToolTip(f"Robots are matched to your clicks only in the first {SEED_WINDOW_S:.0f} seconds.")
        self.frame_slider.sliderReleased.connect(self._frame_changed)
        # Keys, the wheel and track clicks change the value without a release.
        self.frame_slider.valueChanged.connect(self._slider_value)
        top.addWidget(self.frame_slider, 1)
        self.frame_time = widgets.label("0.0 s", role="muted")
        self.frame_slider.valueChanged.connect(lambda value: self.frame_time.setText(f"{value / 10:.1f} s"))
        top.addWidget(self.frame_time)
        body.addLayout(top)

        self.canvas = IdentityCanvas()
        self.canvas.setAccessibleName("First clear frame: click each robot where it touches the floor")
        self.canvas.pressed.connect(self._frame_clicked)
        body.addWidget(self.canvas, 3)
        body.addWidget(widgets.label("Select a robot card, then click that robot on the frame where it touches the floor. "
                                     "Click a marker to select its card.", role="muted", wrap=True))
        cards = QHBoxLayout()
        self.cards = [RobotCard(index) for index in range(ROBOTS_PER_MATCH)]
        for card in self.cards:
            card.siblings = self.cards
            card.changed.connect(self._card_changed)
            card.chosen.connect(self.select)
            card.swap.connect(self.swap)
            cards.addWidget(card, 1)
        body.addLayout(cards)

        bottom = QHBoxLayout()
        self.status = widgets.label("", wrap=True)
        self.status.setAccessibleName("Identity status")
        bottom.addWidget(self.status, 1)
        self.clear_btn = widgets.button("Clear marker", on_click=self.clear_marker)
        self.save_btn = widgets.button("Save identity", "primary", on_click=self.save)
        self.next_btn = widgets.button("Continue to processing ›",
                                       on_click=lambda: self.continue_processing.emit(self.video["id"]))
        for control in (self.clear_btn, self.save_btn, self.next_btn):
            bottom.addWidget(control)
        body.addLayout(bottom)
        self.current = 0
        self.set_project(None, None)

    # ----- project and video -----

    def set_project(self, db, project_id) -> None:
        same = db is self.db and project_id == self.project_id
        self.db, self.project_id = db, project_id
        has = db is not None and project_id is not None
        self.no_project.setVisible(not has)
        self.body.setVisible(has)
        if not same:
            # A different project: nothing from the previous one is kept.
            self.video, self.dirty = None, False
        self.refresh_videos()

    def videos(self) -> list:
        if self.db is None or self.project_id is None:
            return []
        return self.db.connection.execute(
            "SELECT v.id,v.path,v.match_number,v.analysis_status,"
            "(SELECT COUNT(*) FROM robots r WHERE r.video_id=v.id) AS robot_count "
            "FROM videos v WHERE v.project_id=? ORDER BY v.id", (self.project_id,)).fetchall()

    def refresh_videos(self) -> None:
        current = self.video["id"] if self.video else None
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        for row in self.videos():
            if current is not None and int(row["id"]) == int(current):
                self.video = dict(row)  # e.g. processed since it was loaded
            mark = "✓" if row["robot_count"] else "⚑"
            self.video_combo.addItem(f"{mark} {Path(row['path']).name}" + (f" · {row['match_number']}" if row["match_number"] else ""),
                                     int(row["id"]))
        self.video_combo.blockSignals(False)
        if current is not None and self.video_combo.findData(current) >= 0:
            self.video_combo.setCurrentIndex(self.video_combo.findData(current))
            if not self.dirty:
                # Robots may have been saved elsewhere (Set up and process) since they were loaded.
                self._load_saved(int(current))
                self._render()
        elif self.video_combo.count():
            self.load_video(self.video_combo.itemData(0))
        else:
            self.video = None
            self.canvas.placeholder = "Import a video first"
            self.canvas.set_frame(None)
            self._render()

    def _video_chosen(self, index: int) -> None:
        video_id = self.video_combo.itemData(index)
        if self.video and video_id == self.video["id"]:
            return
        if not self.confirm_discard():
            self.video_combo.setCurrentIndex(self.video_combo.findData(self.video["id"]))
            return
        self.load_video(video_id)

    def confirm_discard(self) -> bool:
        if not self.dirty or self.video is None:
            return True
        answer = QMessageBox.question(
            self, "Unsaved robots", f"Save the robots of {Path(self.video['path']).name} first?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def load_video(self, video_id: int) -> bool:
        row = next((row for row in self.videos() if int(row["id"]) == int(video_id)), None)
        if row is None:
            return False
        self.video = dict(row)
        index = self.video_combo.findData(int(video_id))
        if index >= 0:
            self.video_combo.setCurrentIndex(index)
        self._load_saved(int(video_id))
        self.frame_slider.blockSignals(True); self.frame_slider.setValue(0); self.frame_slider.blockSignals(False)
        self.frame_time.setText("0.0 s")
        path = Path(row["path"])
        frame = read_frame(path, 0) if path.is_file() else None
        self.canvas.placeholder = f"{path.name} cannot be read" if frame is None else ""
        self.canvas.set_frame(frame)
        self.dirty = False
        self.current = next((i for i, entry in enumerate(self.entries) if entry.empty), 0)
        self._render()
        return True

    def _load_saved(self, video_id: int) -> None:
        robots = self.db.robots_for_video(video_id)
        entries = []
        for robot in list(robots)[:ROBOTS_PER_MATCH]:
            keys = robot.keys()
            seed = (int(robot["seed_x"]), int(robot["seed_y"])) if robot["seed_x"] is not None and robot["seed_y"] is not None else None
            entries.append(RobotEntry(id=int(robot["id"]), team=str(robot["team_number"]), alliance=robot["alliance"],
                                      start_position=robot["start_position"] if "start_position" in keys else None,
                                      absent=bool(robot["absent"]) if "absent" in keys else seed is None, seed=seed))
        self.entries = entries + [RobotEntry() for _ in range(ROBOTS_PER_MATCH - len(entries))]
        self.saved_entries = [RobotEntry(**vars(entry)) for entry in self.entries]

    def _slider_value(self, _value: int) -> None:
        if not self.frame_slider.isSliderDown():
            self._frame_changed()

    def _frame_changed(self) -> None:
        if self.video is None:
            return
        frame = read_frame(Path(self.video["path"]), self.frame_slider.value() / 10)
        if frame is not None:
            self.canvas.set_frame(frame)
            self._render()

    # ----- editing -----

    def select(self, index: int) -> None:
        self.current = index
        self._render()

    def _frame_clicked(self, x: float, y: float) -> None:
        for index, entry in enumerate(self.entries):
            if entry.seed is not None and not entry.absent:
                point = self.canvas.to_widget(*entry.seed)
                if (point - self.canvas.to_widget(x, y)).manhattanLength() <= 18:
                    self.select(index)
                    return
        entry = self.entries[self.current]
        entry.seed = (int(round(x)), int(round(y)))
        entry.absent = False
        self.dirty = True
        following = next((i for i, other in enumerate(self.entries) if other.seed is None and not other.absent
                          and i != self.current), None)
        if following is not None and self.entries[self.current].complete:
            self.current = following
        self._render()
        self.cards[self.current].team.setFocus()

    def clear_marker(self) -> None:
        self.entries[self.current].seed = None
        self.dirty = True
        self._render()

    def _card_changed(self) -> None:
        for card, entry in zip(self.cards, self.entries):
            card.read(entry)
        self.dirty = True
        self._render()

    def swap(self, first: int, second: int) -> None:
        """Swap the identities of two robots; the markers stay on the robots they mark."""
        a, b = self.entries[first], self.entries[second]
        for attribute in ("team", "alliance", "start_position"):
            value = getattr(a, attribute)
            setattr(a, attribute, getattr(b, attribute))
            setattr(b, attribute, value)
        self.dirty = True
        self._render()

    def _render(self) -> None:
        self.canvas.entries, self.canvas.current = self.entries, self.current
        self.canvas.update()
        for index, (card, entry) in enumerate(zip(self.cards, self.entries)):
            card.show_entry(entry, self.canvas.frame, index == self.current)
        message, tone, savable = identity_state(self.entries)
        has_video = self.video is not None and self.canvas.frame is not None
        if self.video is None:
            message, tone, savable = "Import a video first.", "info", False
        elif has_video and self.dirty:
            message += " Not saved yet."
        theme.set_tone(self.status, tone)
        self.status.setText(message)
        widgets.disable_with_reason(self.save_btn, has_video and savable and self.dirty,
                                    "Nothing new to save" if savable else "Fix the robots marked above first")
        saved_ready = not self.dirty and any(entry.complete and not entry.absent for entry in self.entries)
        widgets.disable_with_reason(self.next_btn, saved_ready, "Save at least one visible robot first")
        widgets.disable_with_reason(self.clear_btn, self.entries[self.current].seed is not None,
                                    "This robot has no marker")

    # ----- saving -----

    def _results_change(self) -> bool:
        """True when saving would make existing results wrong (more than team numbers or positions changed)."""
        if self.video is None or self.db is None:
            return False
        # Read the status now: the video may have been processed since this page loaded it.
        row = self.db.connection.execute("SELECT analysis_status FROM videos WHERE id=?", (int(self.video["id"]),)).fetchone()
        if row is None or row["analysis_status"] != "ready":
            return False
        before = {entry.id: entry for entry in self.saved_entries if entry.id is not None}
        after = [entry for entry in self.entries if entry.complete]
        if {entry.id for entry in after} != set(before):
            return True
        return any(entry.alliance != before[entry.id].alliance or entry.seed != before[entry.id].seed
                   or entry.absent != before[entry.id].absent for entry in after)

    def save(self) -> bool:
        if self.video is None:
            return False
        _message, _tone, savable = identity_state(self.entries)
        if not savable:
            return False
        reprocess = self._results_change()
        if reprocess:
            answer = QMessageBox.question(
                self, "Robots changed after processing",
                "This match was already processed. Changing alliances, markers or which robots are in it makes those "
                "results wrong, so they are hidden until you process the match again. Save anyway?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Save:
                return False
        entries = [entry for entry in self.entries if entry.complete]
        payload = [{"id": entry.id, "team_number": int(entry.team), "alliance": entry.alliance,
                    "start_position": entry.start_position, "absent": entry.absent,
                    "seed_x": entry.seed[0] if entry.seed else None, "seed_y": entry.seed[1] if entry.seed else None}
                   for entry in entries]
        try:
            ids = self.db.save_identity(int(self.video["id"]), payload)
            if reprocess:
                self.db.set_analysis_status(int(self.video["id"]), "unprocessed")
        except ValueError as exc:
            theme.set_tone(self.status, "danger")
            self.status.setText(f"✕ {exc}")
            return False
        for entry, robot_id in zip(entries, ids):
            entry.id = robot_id
        self.dirty = False
        self.saved_entries = [RobotEntry(**vars(entry)) for entry in self.entries]
        self.refresh_videos()
        self.video = dict(next(row for row in self.videos() if int(row["id"]) == int(self.video["id"])))
        self._render()
        self.saved.emit(int(self.video["id"]))
        return True
