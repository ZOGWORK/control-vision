from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2


def create_highlight(video: Path, output: Path, event_time_s: float, ffmpeg_path: str | None = None,
                     before_s: float = 3.0, after_s: float = 3.0) -> tuple[float, float]:
    executable = ffmpeg_path or shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg and add ffmpeg.exe to PATH or set its path in Settings.")
    start = max(0.0, event_time_s - before_s)
    end = event_time_s + after_s
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    copy_cmd = [executable, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}", "-c", "copy", str(output)]
    result = subprocess.run(copy_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        encode_cmd = [executable, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(output)]
        encoded = subprocess.run(encode_cmd, capture_output=True, text=True, check=False)
        if encoded.returncode != 0:
            raise RuntimeError(f"FFmpeg could not create the clip: {(encoded.stderr or result.stderr).strip()[-500:]}")
    return start, end


def thumbnail(path: Path):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
