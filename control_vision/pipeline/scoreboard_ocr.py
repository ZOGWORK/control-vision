from __future__ import annotations

import re
import logging
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def _read_digits(engine, frame: np.ndarray, polygon: list[list[float]]) -> tuple[int | None, float, str]:
    points = np.asarray(polygon, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(points)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return None, 0.0, ""
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0, ""
    # Keep pixels outside a non-rectangular user ROI from contributing digits.
    local_points = points - np.asarray([x0, y0], dtype=np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_points], 255)
    crop = cv2.bitwise_and(crop, crop, mask=mask)
    # UI overlays are often small in 720p streams. Enlarging the ROI helps OCR.
    crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    result = engine(crop)
    raw = result[0] if isinstance(result, tuple) else result
    records = []
    if raw and hasattr(raw[0], "txts"):
        output = raw[0]
        boxes = getattr(output, "boxes", ()) or ()
        for box, text, confidence in zip(boxes, output.txts, output.scores):
            records.append((float(np.asarray(box)[:, 0].mean()), str(text), float(confidence)))
    else:
        for row in raw or []:
            if len(row) >= 3:
                box, text, confidence = row[0], str(row[1]), float(row[2])
                records.append((float(np.asarray(box)[:, 0].mean()), text, confidence))
    records.sort(key=lambda item: item[0])
    # PaddleOCR commonly returns one record per character on large score
    # overlays. Join spatially ordered digits before converting to a score.
    tokens = [(x, re.sub(r"\D", "", text), confidence, text)
              for x, text, confidence in records if re.sub(r"\D", "", text)]
    tokens = [token for token in tokens if len(token[1]) <= 3]
    if not tokens:
        return None, 0.0, " ".join(record[1] for record in records)
    digit_text = "".join(token[1] for token in tokens)
    if len(digit_text) > 3:
        # The user-drawn crop contains extraneous digits. Prefer the most
        # confident OCR token rather than accidentally reading a team number.
        best = max(tokens, key=lambda token: (token[2], len(token[1])))
        digit_text = best[1]
        confidence = best[2]
    else:
        confidence = sum(token[2] for token in tokens) / len(tokens)
    return int(digit_text), confidence, " ".join(record[1] for record in records)


# Fewest frames that must agree before a later, higher reading counts as the
# score having progressed rather than as a misread.
_MIN_PROGRESSION_FRAMES = 3


def _support(items: list[dict]) -> dict[int, float]:
    votes: dict[int, float] = {}
    for item in items:
        votes[int(item["value"])] = votes.get(int(item["value"]), 0.0) + max(0.0, float(item["confidence"]))
    return votes


def _vote(items: list[dict]) -> int:
    """Value with the most confidence-weighted readings; later frames break ties."""
    votes = _support(items)
    last_seen = {int(item["value"]): index for index, item in enumerate(items)}
    return max(votes, key=lambda value: (round(votes[value], 9), last_seen[value]))


def _is_partial(reading: int, value: int) -> bool:
    """True when ``reading`` looks like ``value`` with some digits not read.

    On a results screen that is still fading in over the field, OCR drops
    digits from either end: real footage read a final 180 as 8 and 18, and a
    final 103 as 3. So a reading is partial when its digits are a shorter,
    contiguous run of the higher value's digits (so it is also lower).
    """
    shown, full = str(reading), str(value)
    return len(shown) < len(full) and shown in full


def _revealed_final(valid: list[dict]) -> tuple[int, int] | None:
    """Find a final value that the end of the window rises into through partial readings.

    Looks for the longest run of frames reaching to the end of the window in
    which every reading is either its highest value or a partial reading of
    it. The run must start with a partial reading (the screen is being
    revealed, not a high outlier followed by lower readings), and no partial
    value may be read on as many frames as a stable score needs: a value held
    that long is a real reading, not a passing frame. Returns the final value
    and the index where the run starts, or None.
    """
    found: tuple[int, int] | None = None
    for start in range(len(valid) - 1, -1, -1):
        values = [int(item["value"]) for item in valid[start:]]
        top = max(values)
        if all(value == top or _is_partial(value, top) for value in values):
            found = (top, start)
    if found is None:
        return None
    top, start = found
    if int(valid[start]["value"]) == top:
        return None
    counts: dict[int, int] = {}
    for item in valid:
        counts[int(item["value"])] = counts.get(int(item["value"]), 0) + 1
    partials = {int(item["value"]) for item in valid[start:] if item["value"] != top}
    if any(counts[value] >= _MIN_PROGRESSION_FRAMES for value in partials):
        return None
    return found


def _consensus(samples: list[dict]) -> dict:
    """Vote the proposed score over the sampled frames.

    Every readable frame votes for its value, weighted by OCR confidence, so
    one misread frame cannot outvote many agreeing frames. Scores only rise
    during a match, so a higher value that first appears later in the tail
    window and then holds a weighted majority of every frame from that point
    on (with at least three frames) is treated as the score progressing to its
    final value; earlier lower frames then do not vote against it.

    The final results screen often fades in at the very end of a video, so
    its first frames read with digits missing (180 read as 8, then 18). When
    the window ends by rising through such passing partial readings into a
    higher value (see ``_revealed_final``), that value is proposed and the
    partial readings before it do not vote against it. Partial readings after
    it still count against its agreement, and the confidence stays capped by
    how few frames read the full value, so a short reveal is proposed with
    low confidence and flagged. The result is a proposal only and must be
    reviewed.
    """
    valid = [item for item in samples if item["value"] is not None]
    if not valid:
        return {"value": None, "confidence": 0.0, "agreement": 0.0, "samples": len(samples), "flagged": True}
    winner = _vote(valid)
    window = valid
    first_seen: dict[int, int] = {}
    for index, item in enumerate(valid):
        first_seen.setdefault(int(item["value"]), index)
    for value, index in sorted(first_seen.items(), key=lambda pair: pair[1]):
        if value <= winner:
            continue
        tail = valid[index:]
        if sum(1 for item in tail if item["value"] == value) < _MIN_PROGRESSION_FRAMES:
            continue
        if _vote(tail) == value:
            winner, window = value, tail
    superseded: list[dict] = []
    revealed = _revealed_final(valid)
    if revealed is not None and revealed[0] >= winner:
        winner, start = revealed
        reveal = valid[start:]
        first_full = next(index for index, item in enumerate(reveal) if item["value"] == winner)
        superseded, window = reveal[:first_full], reveal[first_full:]
    votes = _support(window)
    chosen = [item for item in window if item["value"] == winner]
    total = sum(votes.values())
    agreement = votes[winner] / total if total > 0 else len(chosen) / len(window)
    recognition = sum(item["confidence"] for item in chosen) / len(chosen)
    # Recognition confidence is not the same as temporal evidence. Cap the
    # displayed confidence until at least three frames corroborate the score.
    confidence = recognition * agreement * min(1.0, len(chosen) / 3.0)
    superseded_ids = {id(item) for item in superseded}
    outvoted = sorted({int(item["value"]) for item in valid
                       if item["value"] != winner and id(item) not in superseded_ids})
    later_lower = valid[-1]["value"] < winner
    return {"value": int(winner), "confidence": round(confidence, 4),
            "recognition_confidence": round(recognition, 4),
            "agreement": round(agreement, 4), "samples": len(valid),
            "supporting_samples": len(chosen),
            "outvoted_values": outvoted,
            "superseded_partial_values": sorted({int(item["value"]) for item in superseded}),
            "uncorroborated_prior_high": any(value > winner for value in outvoted),
            "later_lower_reading": later_lower,
            # Needs a closer look than an ordinary review: few frames read
            # this value, frames it rests on read something else, or the
            # video ends on a lower reading.
            "flagged": bool(len(chosen) < _MIN_PROGRESSION_FRAMES or len(chosen) < len(window) or later_lower)}


RAPIDOCR_REQUIREMENT = "rapidocr-onnxruntime==1.4.4"
# Stop sampling only after this many frame reads fail in a row; a single
# undecodable frame is skipped and logged.
_MAX_CONSECUTIVE_READ_FAILURES = 5
_MIN_ROI_AREA_PX = 16.0


def _validate_roi(name: str, polygon) -> np.ndarray:
    try:
        points = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The {name} scoreboard area is not valid. Draw it again around the score digits.") from exc
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError(f"The {name} scoreboard area is missing or has fewer than three points. "
                         "Draw it again around the score digits.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"The {name} scoreboard area is not valid. Draw it again around the score digits.")
    if abs(cv2.contourArea(points.astype(np.float32))) < _MIN_ROI_AREA_PX:
        raise ValueError(f"The {name} scoreboard area is too small. Draw it again around the score digits.")
    return points


def _check_roi_in_frame(name: str, points: np.ndarray, frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
        raise ValueError(f"The {name} scoreboard area is outside the video frame ({width}×{height}). "
                         "Draw it again around the score digits.")


def read_scoreboard(video_path: Path, red_roi: list[list[float]], blue_roi: list[list[float]],
                    sample_fps: float = 1.0, tail_seconds: float = 12.0,
                    progress: Callable[[float, str], None] | None = None,
                    cancel: Callable[[], bool] | None = None) -> dict:
    """Read the two alliance score ROIs locally and vote on the final stable score.

    RapidOCR bundles its ONNX OCR models. Results are proposals only; callers
    must keep these separate from confirmed scoring events and official scores.

    The default of one sample per second is deliberate. On a real 720p match
    video each sampled frame took about 3 s to read, so the 12 s tail took
    35-46 s at 1 fps, 73 s at 2 fps and 151 s at 4 fps, and the denser
    sampling caught more frames where OCR drops digits from the clean final
    screen (103 read as 3), which made the blue proposal wrong (3) at both
    2 and 4 fps rather than better.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Scoreboard OCR needs the rapidocr-onnxruntime package, which is not installed. "
                           f"Install it with: python -m pip install {RAPIDOCR_REQUIREMENT}") from exc
    if not math.isfinite(sample_fps) or sample_fps <= 0 or not math.isfinite(tail_seconds) or tail_seconds <= 0:
        raise ValueError("OCR sample rate and tail window must be positive and finite.")
    red_points = _validate_roi("red", red_roi)
    blue_points = _validate_roi("blue", blue_roi)
    logger = logging.getLogger(__name__)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not read video file {video_path.name}.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Could not read video timing from {video_path.name}.")
        duration = frame_count / fps
        stride = max(1, round(fps / sample_fps))
        start_t = max(0.0, duration - tail_seconds)
        start_frame = int(start_t * fps)
        engine = RapidOCR()
        red_samples, blue_samples = [], []
        frame_index = start_frame
        target_count = max(1, (frame_count - start_frame) // stride)
        consecutive_failures = 0
        skipped_frames: list[int] = []
        rois_checked = False
        while frame_index < frame_count:
            if cancel and cancel():
                raise InterruptedError("Scoreboard OCR was cancelled.")
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
            except Exception:
                logger.warning("Could not read scoreboard frame %s in %s", frame_index, video_path.name, exc_info=True)
                ok, frame = False, None
            if not ok or frame is None:
                consecutive_failures += 1
                skipped_frames.append(frame_index)
                logger.warning("Skipped undecodable scoreboard frame %s in %s", frame_index, video_path.name)
                if consecutive_failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                    logger.warning("Stopped scoreboard OCR in %s after %d unreadable frames in a row",
                                   video_path.name, consecutive_failures)
                    break
                frame_index += stride
                continue
            consecutive_failures = 0
            if not rois_checked:
                _check_roi_in_frame("red", red_points, frame)
                _check_roi_in_frame("blue", blue_points, frame)
                rois_checked = True
            t_s = frame_index / fps
            try:
                red_value, red_conf, red_text = _read_digits(engine, frame, red_roi)
                blue_value, blue_conf, blue_text = _read_digits(engine, frame, blue_roi)
            except Exception:
                logger.warning("Skipped unreadable scoreboard frame %s in %s",
                               frame_index, video_path.name, exc_info=True)
                frame_index += stride
                continue
            red_samples.append({"t_s": round(t_s, 3), "value": red_value, "confidence": round(red_conf, 4), "text": red_text})
            blue_samples.append({"t_s": round(t_s, 3), "value": blue_value, "confidence": round(blue_conf, 4), "text": blue_text})
            if progress:
                progress(min(100.0, 100.0 * len(red_samples) / target_count), f"Read scoreboard at {t_s:.1f}s")
            frame_index += stride
    finally:
        cap.release()
    if not red_samples:
        raise RuntimeError(f"Scoreboard OCR could not read any frame in the last {tail_seconds:g} s of "
                           f"{video_path.name}. Check that the video plays to the final results screen.")
    return {"source": "scoreboard_ocr", "window_start_s": round(start_t, 3),
            "frames_sampled": len(red_samples),
            "frames_skipped": len(skipped_frames),
            "red": _consensus(red_samples), "blue": _consensus(blue_samples),
            "red_samples": red_samples, "blue_samples": blue_samples}
