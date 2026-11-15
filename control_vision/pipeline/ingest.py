from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

# Brief: warn (never block) below 720p or 15 fps. Same rule and wording as the
# import dialog in the app so the warning text stays consistent.
MIN_WIDTH, MIN_HEIGHT, MIN_FPS = 1280, 720, 15.0


class VideoError(RuntimeError):
    pass


def probe_video(path: Path) -> dict[str, Any]:
    """Probe with ffprobe; use OpenCV for a fallback when ffprobe is unavailable.

    The result has duration_s, fps, width, height and a "warnings" list of
    plain sentences. Warnings never block an import. The first frame and a
    frame near the end are decoded so corrupt or truncated files are caught.
    """
    metadata = _probe_metadata(path)
    return _check_decodable(path, metadata)


def quality_warnings(metadata: dict[str, Any]) -> list[str]:
    warnings = []
    if metadata["width"] < MIN_WIDTH or metadata["height"] < MIN_HEIGHT:
        warnings.append(f"Resolution is {metadata['width']}\u00d7{metadata['height']}; "
                        f"{MIN_WIDTH}\u00d7{MIN_HEIGHT} or higher is recommended.")
    if metadata["fps"] < MIN_FPS:
        warnings.append(f"Frame rate is {metadata['fps']:.1f} fps; {MIN_FPS:.0f} fps or higher is recommended.")
    return warnings


def _check_decodable(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    warnings = quality_warnings(result)
    fps = float(result["fps"])
    duration = float(result["duration_s"])
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"Could not open video file {path.name}. The file may be corrupt or in an unsupported format.")
        ok, _ = cap.read()
        if not ok:
            raise VideoError(f"Could not decode the first frame of {path.name}. The file may be corrupt or truncated.")
        last_frame = max(0, int(duration * fps) - 1)
        tail_frame = max(0, last_frame - max(1, int(round(fps))))
        if tail_frame > 0 and not _decodes_at(cap, tail_frame):
            result["reported_duration_s"] = duration
            result["truncated"] = True
            # After a failed read OpenCV reports how far decoding got. Verify
            # that position with one more decode instead of searching, which
            # keeps the probe to a few seeks even on long files.
            reached = _position(cap)
            if reached is not None and 1 <= reached < tail_frame and _decodes_at(cap, reached - 1):
                decodable_s = reached / fps
                result["duration_s"] = decodable_s
                warnings.append(f"{path.name} may be truncated: video frames stop decoding near {decodable_s:.1f} s "
                                f"although the file reports {duration:.1f} s.")
            else:
                warnings.append(f"{path.name} may be truncated: frames near the end ({duration:.1f} s) "
                                "could not be decoded.")
    finally:
        cap.release()
    result["warnings"] = warnings
    return result


def _position(cap) -> int | None:
    try:
        value = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
    except (cv2.error, KeyError, TypeError, ValueError):
        return None
    return int(value) if math.isfinite(value) else None


def _decodes_at(cap, frame_index: int) -> bool:
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, _ = cap.read()
    except cv2.error:
        return False
    return bool(ok)


def _probe_metadata(path: Path) -> dict[str, Any]:
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
                fps = _stream_fps(path, stream)
                duration = _positive_float(stream.get("duration"))
                if duration is None:
                    duration = _positive_float(payload.get("format", {}).get("duration"))
                if duration is None and fps > 0:
                    cap = cv2.VideoCapture(str(path))
                    try:
                        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
                    finally:
                        cap.release()
                width, height = int(stream["width"]), int(stream["height"])
                if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0 or duration is None or not math.isfinite(duration) or duration <= 0:
                    raise VideoError(f"Could not read video metadata for {path.name}.")
                return {"duration_s": duration, "fps": fps, "width": width, "height": height}
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
                raise VideoError(f"Could not read video metadata for {path.name}.") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"Could not open video file {path.name}. Check that the file is readable and ffprobe is installed.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width_px = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_px = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        cap.release()
    if not all(math.isfinite(value) and value > 0 for value in (fps, frames, width_px, height_px)):
        raise VideoError(f"Could not read video metadata for {path.name}.")
    return {"duration_s": frames / fps, "fps": fps, "width": int(width_px), "height": int(height_px)}


def _stream_fps(path: Path, stream: dict) -> float:
    """First positive rate of avg_frame_rate, r_frame_rate, then OpenCV.

    ffprobe reports avg_frame_rate as "0/0" for some valid MKV/WebM files.
    """
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = _fraction(str(stream.get(key) or "0/1"))
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(rate) and rate > 0:
            return rate
    cap = cv2.VideoCapture(str(path))
    try:
        rate = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    finally:
        cap.release()
    return rate if math.isfinite(rate) and rate > 0 else 0.0


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def _positive_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None
