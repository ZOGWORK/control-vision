"""Video helpers shared by the Import, Livestream, Calibrate and Identity pages.

Everything here reads from the file itself: duration, resolution, frame rate
and audio come from the container, never from the file size or a guess.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv")
SUPPORTED_TEXT = "MP4, MOV or MKV"
# A recorded match is 2 min 30 s; anything far longer is a stream to split.
LONG_VIDEO_S = 20 * 60


def format_clock(seconds: float) -> str:
    """01:23:45 for long times, 02:30 under an hour."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def parse_clock(text: str) -> float:
    """Seconds from "1:23:45", "23:45", "45" or "45.5"; raises ValueError otherwise."""
    parts = text.strip().split(":")
    if not text.strip() or len(parts) > 3:
        raise ValueError(f"\"{text}\" is not a time. Use h:mm:ss, mm:ss or seconds.")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"\"{text}\" is not a time. Use h:mm:ss, mm:ss or seconds.") from exc
    if any(value < 0 for value in values) or any(value >= 60 for value in values[1:]):
        raise ValueError(f"\"{text}\" is not a time. Use h:mm:ss, mm:ss or seconds.")
    total = 0.0
    for value in values:
        total = total * 60 + value
    return total


def read_frame(path: Path, t_s: float = 0.0) -> np.ndarray | None:
    """The frame at ``t_s`` seconds, or None if it cannot be decoded."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        if t_s > 0:
            capture.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    return frame if ok else None


def ffmpeg_executable(configured: str | None = None) -> str:
    from control_vision.pipeline.clips import _resolve_ffmpeg
    return _resolve_ffmpeg(configured or None)


def has_audio(path: Path, ffmpeg: str | None = None) -> bool | None:
    """Whether the file has an audio stream; None when it cannot be told."""
    try:
        executable = ffmpeg or ffmpeg_executable()
        result = subprocess.run([executable, "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                                timeout=30, encoding="utf-8", errors="replace")
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None
    # ffmpeg exits non-zero without an output file but still prints the streams.
    streams = re.findall(r"Stream #\d+:\d+.*?: (Video|Audio):", result.stderr)
    if not streams:
        return None
    return "Audio" in streams


@dataclass
class FileCheck:
    """What Import shows for one file before it is imported."""
    path: Path
    metadata: dict | None = None
    size: int | None = None
    audio: bool | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.metadata is not None

    @property
    def duration_s(self) -> float | None:
        return float(self.metadata["duration_s"]) if self.metadata else None

    @property
    def resolution(self) -> str:
        return f"{self.metadata['width']}x{self.metadata['height']}" if self.metadata else "—"

    @property
    def vertical(self) -> bool:
        return bool(self.metadata) and self.metadata["height"] > self.metadata["width"]

    @property
    def long(self) -> bool:
        return bool(self.metadata) and float(self.metadata["duration_s"]) > LONG_VIDEO_S


def check_file(path: Path) -> FileCheck:
    """Probe one file. Never raises: a problem is reported in ``error``."""
    path = Path(path)
    check = FileCheck(path=path)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        check.error = f"That file type is not supported. Use {SUPPORTED_TEXT}."
        return check
    try:
        check.size = path.stat().st_size
    except OSError as exc:
        check.error = f"Cannot read the file: {exc}"
        return check
    try:
        from control_vision.pipeline.ingest import probe_video
        check.metadata = probe_video(path)
    except Exception as exc:  # A file that cannot be probed is reported, not imported.
        check.error = f"Cannot read this video: {exc}"
        return check
    check.warnings = list(check.metadata.get("warnings", []))
    if check.vertical:
        check.warnings.append("Vertical — the field may be cropped.")
    check.audio = has_audio(path)
    return check


class _CheckSignals(QObject):
    done = Signal(object)


class _CheckTask(QRunnable):
    def __init__(self, path: Path, signals: _CheckSignals):
        super().__init__()
        self.path, self.signals = path, signals

    def run(self):
        self.signals.done.emit(check_file(self.path))


class FileChecker(QObject):
    """Probes files off the UI thread; ``checked`` fires once per file."""

    checked = Signal(object)

    def __init__(self, parent=None, synchronous: bool = False):
        super().__init__(parent)
        self.synchronous = synchronous
        self._signals = _CheckSignals()
        self._signals.done.connect(self.checked.emit)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)

    def check(self, path: Path) -> None:
        if self.synchronous:
            self.checked.emit(check_file(path))
        else:
            self.pool.start(_CheckTask(Path(path), self._signals))

    def wait(self) -> None:
        self.pool.waitForDone()


def cut_clip(source: Path, start_s: float, end_s: float, output: Path, ffmpeg: str | None = None) -> Path:
    """Copy ``start_s``..``end_s`` of ``source`` into a new file without re-encoding.

    Stream copy starts at the keyframe at or before ``start_s``, so a clip can
    begin a moment early but never late, and the picture is not re-compressed.
    """
    if end_s <= start_s:
        raise ValueError("The end of a match must be after its start.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".part" + output.suffix)
    executable = ffmpeg or ffmpeg_executable()
    command = [executable, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start_s:.3f}", "-i", str(source),
               "-t", f"{end_s - start_s:.3f}", "-map", "0", "-c", "copy", "-avoid_negative_ts", "make_zero",
               str(partial)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, encoding="utf-8",
                                errors="replace")
        if result.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg did not write the clip.")
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output
