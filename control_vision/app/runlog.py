"""What a processing run reports while it runs, read from its progress and its log.

The pipeline hands the app no boxes or positions until a run finishes. While it runs,
it reports progress once per sampled frame and writes a log line per sampled frame
(time and number of robot detections), plus a line whenever a robot's identity is
recovered or re-seeded. These helpers read exactly that, and nothing is guessed.
"""
from __future__ import annotations

import logging
import re

from PySide6.QtCore import QObject, Signal

from control_vision.game.scoreboard import classify_match_time

_OF_TOTAL = re.compile(r"Processed (\d+) of (\d+) sampled frames")
_DONE_ONLY = re.compile(r"Processed (\d+) sampled frames")
_SAMPLED = re.compile(r"Sampled frame (\d+) \(([\d.]+)s\): (\d+) robot detections")
_TIME_AT = re.compile(r"at ([\d.]+) ?s\b")
ID_SWITCH_PREFIXES = ("Recovered robot", "Re-seeded robot")


class RunLogRelay(QObject):
    """Carries log records from the worker thread to the UI thread."""

    line = Signal(str, str)   # level name, message


class RunLogHandler(logging.Handler):
    def __init__(self, relay: RunLogRelay):
        super().__init__(logging.INFO)
        self.relay = relay

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.relay.line.emit(record.levelname, record.getMessage())
        except Exception:  # Never let the UI break a run.
            self.handleError(record)


def frames_progress(percent: float, detail: str) -> tuple[int, int] | None:
    """Sampled frames done and in total, from a progress report; None when it does not say."""
    match = _OF_TOTAL.search(detail or "")
    if match:
        return int(match.group(1)), int(match.group(2))
    match = _DONE_ONLY.search(detail or "")
    if match and percent > 0:
        done = int(match.group(1))
        return done, max(done, round(done * 100 / percent))
    return None


def progress_time(detail: str) -> float | None:
    """The video time a progress report names ("… at 12.5 s"), if any."""
    match = _TIME_AT.search(detail or "")
    return float(match.group(1)) if match else None


def sampled_frame(message: str) -> tuple[int, float, int] | None:
    """(frame, video time in s, robot detections) from a per-frame log line."""
    match = _SAMPLED.search(message or "")
    return (int(match.group(1)), float(match.group(2)), int(match.group(3))) if match else None


def is_id_switch(message: str) -> bool:
    """A robot was given a new track: recovered by re-linking, or re-seeded."""
    return (message or "").startswith(ID_SWITCH_PREFIXES) or "reacquired" in (message or "")


def time_left(elapsed_s: float, percent: float) -> float | None:
    """Seconds left at the current rate; None until there is enough to go on."""
    if percent <= 1 or elapsed_s <= 0:
        return None
    return max(0.0, elapsed_s * (100 - percent) / percent)


class PhaseTracker:
    """Match phase from video time, assuming the video starts when the match starts.

    Reports each change, and flags a jump that skips TRANSITION, which was a real bug once.
    """

    ORDER = ("AUTO", "TRANSITION", "TELEOP", "POST-MATCH")

    def __init__(self, periods: dict):
        self.periods = periods
        self.phase: str | None = None

    def update(self, t_s: float) -> tuple[str, float, str | None, bool]:
        """(phase, clock left in it, change message or None, whether the change skipped a phase)."""
        phase, remaining = classify_match_time(t_s, self.periods)
        message, skipped = None, False
        if self.phase is not None and phase != self.phase:
            message = f"Phase changed: {self.phase} → {phase}"
            skipped = (self.ORDER.index(phase) - self.ORDER.index(self.phase)) > 1
            if skipped:
                missing = self.ORDER[self.ORDER.index(self.phase) + 1]
                message += f" — {missing} missing"
        self.phase = phase
        return phase, remaining, message, skipped
