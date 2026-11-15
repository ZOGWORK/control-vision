from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path

import cv2


FFMPEG_DOWNLOAD_URL = "https://ffmpeg.org/download.html"
# A six-second highlight takes seconds; a hung FFmpeg must not block the app.
_FFMPEG_TIMEOUT_S = 120.0


def _resolve_ffmpeg(ffmpeg_path: str | None) -> str:
    """Return a runnable FFmpeg executable or raise one plain-language error."""
    if ffmpeg_path:
        configured = Path(ffmpeg_path)
        if configured.is_file():
            return str(configured)
        on_path = shutil.which(ffmpeg_path)
        if on_path:
            return on_path
        raise RuntimeError(
            f"The FFmpeg path set in Settings does not exist: {ffmpeg_path}. "
            f"Choose the ffmpeg.exe file again, or download FFmpeg from {FFMPEG_DOWNLOAD_URL}."
        )
    executable = shutil.which("ffmpeg")
    if not executable:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            executable = None
    if not executable:
        raise RuntimeError(
            f"FFmpeg was not found. Download it from {FFMPEG_DOWNLOAD_URL}, "
            "then add ffmpeg.exe to PATH or set its path in Settings."
        )
    return executable


def _video_duration(video: Path) -> float | None:
    """Best-effort container duration from OpenCV; None when it cannot be read."""
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    if math.isfinite(fps) and math.isfinite(frames) and fps > 0 and frames > 0:
        return float(frames / fps)
    return None


def _run(command: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False, timeout=_FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg did not finish the clip within {_FFMPEG_TIMEOUT_S:.0f} seconds.") from exc
    except OSError as exc:
        raise RuntimeError(
            f"FFmpeg could not be started from {command[0]}. "
            f"Check the FFmpeg path in Settings or download FFmpeg from {FFMPEG_DOWNLOAD_URL}."
        ) from exc


def create_highlight(video: Path, output: Path, event_time_s: float, ffmpeg_path: str | None = None,
                     before_s: float = 3.0, after_s: float = 3.0, *,
                     duration_s: float | None = None) -> tuple[float, float]:
    """Cut event_time_s - before_s .. event_time_s + after_s, clamped to the video.

    The returned (start, end) describe the clip that was written. When
    duration_s is not given it is read from the video when possible.
    """
    if not math.isfinite(event_time_s) or event_time_s < 0:
        raise ValueError(f"The event time {event_time_s} s is not a valid time in {video.name}.")
    if duration_s is None:
        duration_s = _video_duration(video)
    if duration_s is not None:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError(f"The video duration for {video.name} is not valid.")
        if event_time_s > duration_s:
            raise ValueError(
                f"The event at {event_time_s:.3f} s is after the end of {video.name} ({duration_s:.3f} s)."
            )
    executable = _resolve_ffmpeg(ffmpeg_path)
    start = max(0.0, event_time_s - before_s)
    end = event_time_s + after_s
    if duration_s is not None:
        end = min(end, duration_s)
    if end <= start:
        raise ValueError(f"The clip window for the event at {event_time_s:.3f} s in {video.name} is empty.")
    output.parent.mkdir(parents=True, exist_ok=True)
    length = end - start
    copy_cmd = [executable, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{length:.3f}", "-c", "copy", str(output)]
    try:
        result = _run(copy_cmd)
    except RuntimeError:
        output.unlink(missing_ok=True)
        raise
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        encode_cmd = [executable, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{length:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(output)]
        try:
            encoded = _run(encode_cmd)
        except RuntimeError:
            output.unlink(missing_ok=True)
            raise
        if encoded.returncode != 0:
            output.unlink(missing_ok=True)
            logging.getLogger(__name__).warning("FFmpeg failed for %s: %s", video.name,
                                                (encoded.stderr or result.stderr or "").strip()[-2000:])
            raise RuntimeError(f"FFmpeg could not create the clip: {(encoded.stderr or result.stderr).strip()[-500:]}")
    return start, end


def thumbnail(path: Path):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
