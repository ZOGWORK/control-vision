from __future__ import annotations

import re
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


def _consensus(samples: list[dict]) -> dict:
    valid = [item for item in samples if item["value"] is not None]
    if not valid:
        return {"value": None, "confidence": 0.0, "agreement": 0.0, "samples": len(samples)}
    # Scores are monotonic during ordinary match play. Keep the largest
    # observed reading as a proposal, but lower its confidence when the final
    # samples do not corroborate it; a reviewer must accept it explicitly.
    recent = valid[-3:]
    winner = max(item["value"] for item in valid)
    chosen = [item for item in recent if item["value"] == winner]
    agreement = len(chosen) / len(recent)
    evidence = chosen or [next(item for item in reversed(valid) if item["value"] == winner)]
    recognition = sum(item["confidence"] for item in evidence) / len(evidence)
    # Recognition confidence is not the same as temporal evidence. Cap the
    # displayed confidence until at least three frames corroborate the score.
    confidence = recognition * agreement * min(1.0, len(recent) / 3.0)
    return {"value": int(winner), "confidence": round(confidence, 4),
            "recognition_confidence": round(recognition, 4),
            "agreement": round(agreement, 4), "samples": len(valid),
            "uncorroborated_prior_high": not chosen,
            "later_lower_reading": valid[-1]["value"] < winner}


def read_scoreboard(video_path: Path, red_roi: list[list[float]], blue_roi: list[list[float]],
                    sample_fps: float = 1.0, tail_seconds: float = 12.0,
                    progress: Callable[[float, str], None] | None = None,
                    cancel: Callable[[], bool] | None = None) -> dict:
    """Read the two alliance score ROIs locally and vote on the final stable score.

    RapidOCR bundles its ONNX OCR models. Results are proposals only; callers
    must keep these separate from confirmed scoring events and official scores.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Local RapidOCR models are not installed. Reinstall the app with the pinned requirements.") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read video file {video_path.name}.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video timing from {video_path.name}.")
    duration = frame_count / fps
    stride = max(1, round(fps / max(0.1, sample_fps)))
    start_t = max(0.0, duration - tail_seconds)
    start_frame = int(start_t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    engine = RapidOCR()
    red_samples, blue_samples = [], []
    frame_index = start_frame
    target_count = max(1, (frame_count - start_frame) // stride)
    try:
        while frame_index < frame_count:
            if cancel and cancel():
                raise InterruptedError("Scoreboard OCR was cancelled.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            t_s = frame_index / fps
            red_value, red_conf, red_text = _read_digits(engine, frame, red_roi)
            blue_value, blue_conf, blue_text = _read_digits(engine, frame, blue_roi)
            red_samples.append({"t_s": round(t_s, 3), "value": red_value, "confidence": round(red_conf, 4), "text": red_text})
            blue_samples.append({"t_s": round(t_s, 3), "value": blue_value, "confidence": round(blue_conf, 4), "text": blue_text})
            if progress:
                progress(min(100.0, 100.0 * len(red_samples) / target_count), f"Read scoreboard at {t_s:.1f}s")
            frame_index += stride
    finally:
        cap.release()
    return {"source": "scoreboard_ocr", "window_start_s": round(start_t, 3),
            "frames_sampled": len(red_samples),
            "red": _consensus(red_samples), "blue": _consensus(blue_samples),
            "red_samples": red_samples, "blue_samples": blue_samples}
