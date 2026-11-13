from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2


class VideoError(RuntimeError):
    pass


def probe_video(path: Path) -> dict[str, float | int]:
    """Probe with ffprobe; use OpenCV for a fallback when ffprobe is unavailable."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout)
                stream = payload["streams"][0]
                fps = _fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
                duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
                if duration <= 0 and fps > 0:
                    cap = cv2.VideoCapture(str(path))
                    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
                    cap.release()
                return {"duration_s": duration, "fps": fps, "width": int(stream["width"]), "height": int(stream["height"])}
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
                raise VideoError(f"Could not read video metadata for {path.name}.") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video file {path.name}. Check that the file is readable and ffprobe is installed.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise VideoError(f"Could not read video metadata for {path.name}.")
    return {"duration_s": frames / fps, "fps": fps, "width": width, "height": height}


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0
