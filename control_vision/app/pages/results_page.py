"""Match results overview: who won, by how much, and how our count compares (guide page 15).

Our totals come from the confirmed scoring events, scored with the same rules as Score and
export. The official score is whatever the user entered (or took from the scoreboard OCR),
shown next to ours and never merged. Nothing is final while events still need review.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from control_vision.app import theme, widgets
from control_vision.scout.exports import match_scoreboard

OFFICIAL_TOLERANCE = 0.10     # the guide's ±10% check
LIVE_NOTE = ("End-of-match points come from the results screen, not the live video, so a total read from the video "
             "can trail the official one.")


def official_check(ours: int, official: int | None) -> tuple[str, str]:
    """Text and tone comparing our alliance total with the official one."""
    if official is None:
        return "Official score not entered yet", "pending"
    if official == 0:
        return ("Matches official", "success") if ours == 0 else ("Official is 0 — check Review", "danger")
    difference = abs(ours - official) / official
    if difference <= OFFICIAL_TOLERANCE:
        return f"Within ±10% of official ({ours - official:+d})", "success"
    return f"Differs from official by {difference:.0%} — check Review", "danger"


def event_alliance(event) -> str | None:
    """The alliance an event scores for, in the scoreboard's order: scoring alliance, event alliance, robot."""
    keys = event.keys()
    for key in ("scoring_alliance", "event_alliance", "alliance"):
        if key in keys and event[key]:
            return event[key]
    return None


class ResultsPage(QWidget):
    open_map = Signal(int)
    open_timeline = Signal(int)
    open_highlights = Signal(int)
    open_review = Signal(int)
    open_queue = Signal()
    official_saved = Signal(int)
    new_project = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")
        self.db = None
        self.project_id = None
        self.video: dict | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        self.no_project = widgets.EmptyState("Open a project to see match results", "", "+ New project",
                                             self.new_project.emit)
        outer.addWidget(self.no_project, 1)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        outer.addWidget(self.body, 1)

        top = QHBoxLayout()
        match_label = QLabel("&Match")
        self.video_combo = QComboBox(); self.video_combo.setMinimumWidth(320)
        self.video_combo.setAccessibleName("Match")
        match_label.setBuddy(self.video_combo)
        self.video_combo.currentIndexChanged.connect(lambda _i: self.show_video(self.video_combo.currentData()))
        top.addWidget(match_label); top.addWidget(self.video_combo); top.addStretch(1)
        self.review_badge = widgets.label("")
        top.addWidget(self.review_badge)
        body.addLayout(top)

        self.not_processed = widgets.EmptyState("Processing not finished",
                                                "Results appear here once this match has been processed.",
                                                "Open the processing queue", self.open_queue.emit)
        body.addWidget(self.not_processed, 1)

        self.results = QWidget()
        results = QVBoxLayout(self.results)
        results.setContentsMargins(0, 0, 0, 0)
        results.setSpacing(12)
        body.addWidget(self.results, 1)
        self.header = widgets.label("", role="section", wrap=True)
        results.addWidget(self.header)

        scores = QHBoxLayout()
        self.cards = {}
        for alliance in ("red", "blue"):
            card = widgets.card()
            card.setProperty("alliance", alliance)
            grid = QGridLayout(card)
            grid.setContentsMargins(16, 12, 16, 12)
            grid.addWidget(widgets.label(f"{alliance.title()} alliance", role="section"), 0, 0, 1, 2)
            ours = widgets.label("—"); ours.setProperty("role", "metric")
            ours.setAccessibleName(f"{alliance.title()} total, ours")
            grid.addWidget(widgets.label("Ours", role="muted"), 1, 0); grid.addWidget(ours, 2, 0)
            official = QSpinBox(); official.setRange(-1, 999); official.setSpecialValueText("Not entered")
            official.setAccessibleName(f"{alliance.title()} official score")
            grid.addWidget(widgets.label("Official", role="muted"), 1, 1); grid.addWidget(official, 2, 1)
            check = widgets.label("", wrap=True)
            grid.addWidget(check, 3, 0, 1, 2)
            detail = widgets.label("", role="muted", wrap=True)
            grid.addWidget(detail, 4, 0, 1, 2)
            self.cards[alliance] = {"ours": ours, "official": official, "check": check, "detail": detail}
            scores.addWidget(card, 1)
        results.addLayout(scores)
        official_row = QHBoxLayout()
        self.save_official_btn = widgets.button("Save official score", on_click=self.save_official,
                                                tooltip="From ftc-events.firstinspires.org or the results screen.")
        official_row.addWidget(widgets.label(LIVE_NOTE, role="muted", wrap=True), 1)
        official_row.addWidget(self.save_official_btn)
        results.addLayout(official_row)

        results.addWidget(widgets.section_label("Robots"))
        self.robot_row = QHBoxLayout()
        results.addLayout(self.robot_row)
        self.accuracy = widgets.label("", wrap=True)
        results.addWidget(self.accuracy)

        buttons = QHBoxLayout()
        for text, signal in (("View map", self.open_map), ("View timeline", self.open_timeline),
                             ("View highlights", self.open_highlights), ("Open review queue", self.open_review)):
            buttons.addWidget(widgets.button(text, on_click=lambda _=False, s=signal: s.emit(int(self.video["id"]))))
        buttons.addStretch(1)
        results.addLayout(buttons)
        results.addStretch(1)
        self.no_project.show(); self.body.hide()

    # ----- project and match -----

    def set_project(self, db, project_id) -> None:
        same = db is self.db and project_id == self.project_id
        self.db, self.project_id = db, project_id
        has = db is not None and project_id is not None
        self.no_project.setVisible(not has)
        self.body.setVisible(has)
        if not same:
            self.video = None
        self.refresh_videos()

    def refresh_videos(self, select: int | None = None) -> None:
        current = select if select is not None else (self.video["id"] if self.video else None)
        rows = [] if self.db is None else self.db.connection.execute(
            "SELECT id,path,match_number,analysis_status FROM videos WHERE project_id=? ORDER BY id",
            (self.project_id,)).fetchall()
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        for row in rows:
            mark = "✓" if row["analysis_status"] == "ready" else "○"
            self.video_combo.addItem(f"{mark} {Path(row['path']).name}" + (f" · {row['match_number']}" if row["match_number"] else ""),
                                     int(row["id"]))
        index = self.video_combo.findData(int(current)) if current is not None else -1
        if index < 0:
            index = next((i for i, row in enumerate(rows) if row["analysis_status"] == "ready"), 0 if rows else -1)
        self.video_combo.setCurrentIndex(index)
        self.video_combo.blockSignals(False)
        self.show_video(self.video_combo.currentData())

    def show_video(self, video_id) -> None:
        if self.db is None or video_id is None:
            self.video = None
            self.not_processed.show(); self.results.hide(); self.review_badge.setText("")
            return
        row = self.db.connection.execute("SELECT * FROM videos WHERE id=?", (int(video_id),)).fetchone()
        if row is None:
            return
        self.video = dict(row)
        ready = self.video["analysis_status"] == "ready"
        self.not_processed.setVisible(not ready)
        self.results.setVisible(ready)
        if ready:
            self._render()
        else:
            self.review_badge.setText("")

    # ----- rendering -----

    def board(self, robots, events) -> dict:
        return match_scoreboard(self.video, [dict(robot) for robot in robots], [dict(event) for event in events])

    def _render(self) -> None:
        video_id = int(self.video["id"])
        robots = list(self.db.robots_for_video(video_id))
        events = list(self.db.events_for_video(video_id))
        cycles = list(self.db.cycles_for_video(video_id))
        board = self.board(robots, events)
        flagged = [event for event in events if event["status"] == "flagged"]
        teams = {alliance: [str(robot["team_number"]) for robot in robots if robot["alliance"] == alliance]
                 for alliance in ("red", "blue")}
        parts = [Path(self.video["path"]).name, self.video["event_code"] or "Practice", self.video["match_number"] or ""]
        self.header.setText(" · ".join(part for part in parts if part) +
                            f"   Red {' + '.join(teams['red']) or '—'}  vs  Blue {' + '.join(teams['blue']) or '—'}")
        if flagged:
            self.review_badge.setText(f"⚑ Not reviewed yet — {len(flagged)} event(s) need review; score is not final")
            theme.set_tone(self.review_badge, "pending")
        else:
            self.review_badge.setText("✓ Reviewed — no events need review")
            theme.set_tone(self.review_badge, "success")
        confirmed = [event for event in events if event["status"] == "confirmed"]
        for alliance, card in self.cards.items():
            ours = int(board["alliance_points"][alliance])
            official = self.video[f"official_{alliance}"]
            card["ours"].setText(str(ours))
            card["official"].blockSignals(True)
            card["official"].setValue(-1 if official is None else int(official))
            card["official"].blockSignals(False)
            text, tone = official_check(ours, None if official is None else int(official))
            card["check"].setText(text)
            theme.set_tone(card["check"], tone)
            tips = board["ranking_points"][alliance].get("hive_tips", 0) if "ranking_points" in board else 0
            counted = sum(1 for event in confirmed if event_alliance(event) == alliance)
            card["detail"].setText(f"{counted} confirmed scoring event(s) · Hive tips {tips}")
        self._render_robots(robots, board, cycles, flagged)
        self.accuracy.setText("Accuracy: within ±1 cycle of a hand count — Not yet checked (no hand count is stored "
                              "for this match). Ours vs official is shown on each alliance card.")

    def _render_robots(self, robots, board, cycles, flagged) -> None:
        while self.robot_row.count():
            item = self.robot_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Detach now: deleteLater alone leaves the old widget painted until the event loop runs.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        points = board.get("team_points", {})
        for robot in robots:
            robot_id = int(robot["id"])
            card = widgets.card()
            card.setProperty("alliance", robot["alliance"])
            absent = bool(robot["absent"]) if "absent" in robot.keys() else False
            if absent:
                card.setProperty("state", "absent")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.addWidget(widgets.label(f"{robot['team_number']} · {robot['alliance'].title()}", role="section"))
            if absent:
                layout.addWidget(widgets.label("Absent · 0 points", role="muted"))
            else:
                own = [cycle for cycle in cycles if int(cycle["robot_id"]) == robot_id]
                review = sum(1 for event in flagged if event["robot_id"] is not None and int(event["robot_id"]) == robot_id)
                layout.addWidget(widgets.label(f"{int(points.get(robot_id, 0))} points attributed"))
                layout.addWidget(widgets.label(f"{len(own)} cycle(s)", role="muted"))
                if review:
                    layout.addWidget(widgets.label(f"⚑ {review} need review", tone="pending"))
            self.robot_row.addWidget(card, 1)
        self.robot_row.addStretch(0)

    # ----- official score -----

    def save_official(self) -> None:
        if self.video is None:
            return
        values = {alliance: card["official"].value() for alliance, card in self.cards.items()}
        self.db.set_official_scores(
            int(self.video["id"]), None if values["red"] < 0 else values["red"], None if values["blue"] < 0 else values["blue"],
            swarm_threshold=int(self.video["swarm_threshold"] or 16),
            pollinator_1_threshold=int(self.video["pollinator_1_threshold"] or 4),
            pollinator_2_threshold=int(self.video["pollinator_2_threshold"] or 7))
        self.show_video(int(self.video["id"]))
        self.official_saved.emit(int(self.video["id"]))
