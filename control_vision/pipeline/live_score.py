"""Read the live score overlay through a match and propose timestamped scoring events.

FIRST Tech Challenge event streams show a live overlay (FTC Live) with each
alliance's running score, the match clock and each alliance's HIVE TIP count
("2 /4"). Scorekeepers enter HIVE TIPS and foul credits as they happen, so the
overlay gives the time and point value of those achievements; LEAVE, PARK and
the end-state CELL, FLOWER and GARDEN counts are entered at the end of a period.

This module locates the overlay, reads it at a fixed sample rate, and turns
the readings into candidate scoring events. Every event is a proposal: it is
flagged for human review and never confirmed here. The overlay can show
scorekeeper corrections, a TIP and a MAJOR FOUL credit are both 20 points, and
end-state points are only known after the match, so a reviewer must check
each event against the video and the official result.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in frame pixels

# Overlay areas relative to the "Red" and "Blue" alliance labels, in units of
# the horizontal distance between the two label centres (d). Measured on the
# standard FTC Live overlay at two events with the overlay at the top and at
# the bottom of a 1920x1080 stream; the layout is the same in both.
# (x offset from the anchor label centre, y offset from the label centre) for
# the two corners of each area.
_LAYOUT = {
    "red_score": ("red", (-0.20, 0.06), (0.20, 0.30)),
    "blue_score": ("blue", (-0.20, 0.06), (0.20, 0.30)),
    # Bands that contain the TIP counter. The counter moves between AUTO and
    # TELEOP (it is drawn further from the team numbers in AUTO), so the band
    # spans both positions and the counter's white box is found inside it.
    "red_tips": ("red", (-1.80, 0.0), (-0.85, 0.25)),
    "blue_tips": ("blue", (0.85, 0.0), (1.80, 0.25)),
    "timer": ("middle", (-0.26, 0.0), (0.26, 0.21)),
}
# The two labels are on one row: their centres differ in height by less than
# this fraction of d.
_LABEL_ROW_TOLERANCE = 0.1


@dataclass(frozen=True)
class OverlayLayout:
    """Frame areas of the live overlay fields."""
    red_score: Box
    blue_score: Box
    red_tips: Box
    blue_tips: Box
    timer: Box

    def as_dict(self) -> dict[str, list[int]]:
        return {name: list(getattr(self, name)) for name in _LAYOUT}


def layout_from_labels(red_label: Box, blue_label: Box, frame_shape: tuple[int, ...]) -> OverlayLayout | None:
    """Overlay areas from the boxes of the "Red" and "Blue" labels, or None."""
    red_cx, red_cy = (red_label[0] + red_label[2]) / 2, (red_label[1] + red_label[3]) / 2
    blue_cx, blue_cy = (blue_label[0] + blue_label[2]) / 2, (blue_label[1] + blue_label[3]) / 2
    d = blue_cx - red_cx
    if d <= 0 or abs(blue_cy - red_cy) > _LABEL_ROW_TOLERANCE * d:
        return None
    anchors = {"red": (red_cx, red_cy), "blue": (blue_cx, blue_cy),
               "middle": ((red_cx + blue_cx) / 2, (red_cy + blue_cy) / 2)}
    height, width = frame_shape[:2]
    boxes = {}
    for name, (anchor, (ax0, ay0), (ax1, ay1)) in _LAYOUT.items():
        cx, cy = anchors[anchor]
        x0, y0 = max(0, int(cx + ax0 * d)), max(0, int(cy + ay0 * d))
        x1, y1 = min(width, int(math.ceil(cx + ax1 * d))), min(height, int(math.ceil(cy + ay1 * d)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        boxes[name] = (x0, y0, x1, y1)
    return OverlayLayout(**boxes)


def _label_boxes(result) -> tuple[Box | None, Box | None]:
    """The most confident "Red" and "Blue" text boxes in a full-frame OCR result."""
    raw = result[0] if isinstance(result, tuple) else result
    best: dict[str, tuple[float, Box]] = {}
    for row in raw or []:
        if len(row) < 3:
            continue
        text = re.sub(r"\W", "", str(row[1])).lower()
        if text not in {"red", "blue"}:
            continue
        points = np.asarray(row[0], dtype=np.float64).reshape(-1, 2)
        box = (int(points[:, 0].min()), int(points[:, 1].min()), int(points[:, 0].max()), int(points[:, 1].max()))
        if text not in best or float(row[2]) > best[text][0]:
            best[text] = (float(row[2]), box)
    return (best["red"][1] if "red" in best else None, best["blue"][1] if "blue" in best else None)


def locate_overlay(engine, frame: np.ndarray) -> OverlayLayout | None:
    """Find the live overlay in one frame by its "Red" and "Blue" labels."""
    red, blue = _label_boxes(engine(frame))
    if red is None or blue is None:
        return None
    return layout_from_labels(red, blue, frame.shape)


_TIMER_RE = re.compile(r"(\d{1,2})\s*[:.;]\s*(\d{2})")


def parse_clock(text: str) -> int | None:
    """Seconds shown by a "M:SS" clock reading, or None."""
    match = _TIMER_RE.search(text or "")
    if not match:
        return None
    minutes, seconds = int(match.group(1)), int(match.group(2))
    if seconds >= 60 or minutes > 2:
        return None
    return minutes * 60 + seconds


_ZERO_LOOKALIKES = str.maketrans({"。": "0", "O": "0", "o": "0"})


def parse_score(text: str) -> int | None:
    """An alliance score reading (up to three digits, no leading zero), or None.

    A lone large "0" is sometimes recognised as a round symbol, so those are
    read as zero.
    """
    digits = re.sub(r"\D", "", (text or "").translate(_ZERO_LOOKALIKES))
    if not digits or len(digits) > 3 or (len(digits) > 1 and digits[0] == "0"):
        return None
    return int(digits)


def parse_tips(text: str) -> int | None:
    """The HIVE TIP count read from the counter's white box, or None."""
    digits = re.sub(r"\D", "", (text or "").translate(_ZERO_LOOKALIKES))
    if not digits or len(digits) > 2:
        return None
    return int(digits)


def _crop(frame: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1]


# The TIP counter is dark digits in a solid white box. Within the band, a
# white region counts as that box when it is mostly filled (outline icons
# are not), about as tall as it is wide or a little wider, and at least this
# fraction of the band's height.
_WHITE_MIN_VALUE = 200
_WHITE_MAX_SATURATION = 60
_COUNTER_MIN_FILL = 0.6
_COUNTER_MIN_HEIGHT = 0.3
_COUNTER_ASPECT = (0.6, 2.2)
_DARK_MAX_VALUE = 110
_COUNTER_DARK_SHARE = (0.03, 0.6)


def counter_box(band: np.ndarray, inner_side: str) -> Box | None:
    """The TIP counter's white box inside a band crop, in band pixels, or None.

    In TELEOP the overlay also shows a white panel of robot check marks
    further out, so of the qualifying boxes the one nearest the team numbers
    (``inner_side``, "right" for red and "left" for blue) is the counter.
    """
    if band.size == 0:
        return None
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] >= _WHITE_MIN_VALUE) & (hsv[:, :, 1] <= _WHITE_MAX_SATURATION)).astype(np.uint8)
    # The box's white surrounds its dark digits, so it is one connected
    # region on its own. (Closing the gaps would merge it with the white
    # "/4" text next to it.)
    count, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    best: Box | None = None
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if h < _COUNTER_MIN_HEIGHT * band.shape[0] or not _COUNTER_ASPECT[0] <= w / h <= _COUNTER_ASPECT[1]:
            continue
        if area / float(w * h) < _COUNTER_MIN_FILL:
            continue
        # The counter has dark digits in it; a white glyph or icon closed
        # into a solid region has only coloured or white pixels.
        dark = float((hsv[y:y + h, x:x + w, 2] < _DARK_MAX_VALUE).mean())
        if not _COUNTER_DARK_SHARE[0] <= dark <= _COUNTER_DARK_SHARE[1]:
            continue
        box = (x, y, x + w, y + h)
        if best is None or (box[2] > best[2] if inner_side == "right" else box[0] < best[0]):
            best = box
    return best


def read_frame(recognizer, frame: np.ndarray, layout: OverlayLayout) -> dict:
    """Read every overlay field of one frame with one batched recognition pass."""
    names = list(_LAYOUT)
    crops = []
    for name in names:
        crop = _crop(frame, getattr(layout, name))
        if name.endswith("_tips"):
            inner = counter_box(crop, "right" if name == "red_tips" else "left")
            # No white counter box in the band: recognise an empty crop so
            # the field reads as missing rather than as another number.
            crop = crop[inner[1]:inner[3], inner[0]:inner[2]] if inner else np.full((8, 8, 3), 255, np.uint8)
        crops.append(crop)
    results, _ = recognizer(crops)
    row: dict = {}
    for name, (text, confidence) in zip(names, results):
        row[name + "_text"] = str(text)
        row[name + "_conf"] = round(float(confidence), 4)
    row["red_score"] = parse_score(row["red_score_text"])
    row["blue_score"] = parse_score(row["blue_score_text"])
    row["red_tips"] = parse_tips(row["red_tips_text"])
    row["blue_tips"] = parse_tips(row["blue_tips_text"])
    row["clock_s"] = parse_clock(row["timer_text"])
    return row


RAPIDOCR_REQUIREMENT = "rapidocr-onnxruntime==1.4.4"
_LOCATE_FRACTIONS = (0.5, 0.35, 0.65, 0.2, 0.8)
_MAX_CONSECUTIVE_READ_FAILURES = 5


def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Reading the live score overlay needs the rapidocr-onnxruntime package, which is not "
                           f"installed. Install it with: python -m pip install {RAPIDOCR_REQUIREMENT}") from exc
    return RapidOCR()


def _decode(cap, logger, video_path: Path, index: int, grab: bool = False):
    """(ok, frame) for the next frame; a decoder that raises counts as a failed decode."""
    try:
        if grab:
            return bool(cap.grab()), None
        ok, frame = cap.read()
        return bool(ok) and frame is not None, frame
    except Exception:
        logger.warning("The decoder raised on frame %s of %s", index, video_path.name, exc_info=True)
        return False, None


def read_live_overlay(video_path: Path, layout: OverlayLayout | None = None, sample_fps: float = 2.0,
                      progress: Callable[[float, str], None] | None = None,
                      cancel: Callable[[], bool] | None = None) -> dict:
    """Read the live overlay through the whole video.

    Frames are decoded in order (grabbing the ones between samples), which
    is much faster than seeking to each sample. Returns the layout used and
    one reading per sampled frame.
    """
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("The overlay sample rate must be positive and finite.")
    logger = logging.getLogger(__name__)
    engine = _engine()
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not read video file {video_path.name}.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Could not read video timing from {video_path.name}.")
        if layout is None:
            for fraction in _LOCATE_FRACTIONS:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fraction * frame_count))
                ok, frame = _decode(cap, logger, video_path, int(fraction * frame_count))
                if ok:
                    try:
                        layout = locate_overlay(engine, frame)
                    except Exception:
                        # One bad probe frame must not stop the other probes.
                        logger.warning("Could not search frame %d of %s for the overlay",
                                       int(fraction * frame_count), video_path.name, exc_info=True)
                        layout = None
                    if layout is not None:
                        break
            if layout is None:
                raise RuntimeError(f"Could not find the live score overlay in {video_path.name}. The video needs "
                                   "the FTC Live overlay with the \"Red\" and \"Blue\" alliance labels.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != 0:
                # Timestamps count frames from the start, so a backend that
                # refuses to rewind must not continue from the probe position.
                logger.warning("Could not rewind %s after locating the overlay; reopening it", video_path.name)
                cap.release()
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not read video file {video_path.name}.")
        stride = max(1, round(fps / sample_fps))
        samples: list[dict] = []
        failures = 0
        index = 0
        stopped: str | None = None
        while index < frame_count:
            if cancel and cancel():
                raise InterruptedError("Reading the live score overlay was cancelled.")
            if index % stride:
                if not _decode(cap, logger, video_path, index, grab=True)[0]:
                    stopped = f"the video could not be decoded past {index / fps:.1f} s"
                    break
                index += 1
                continue
            ok, frame = _decode(cap, logger, video_path, index)
            index += 1
            row = None
            if ok:
                try:
                    row = read_frame(engine.text_rec, frame, layout)
                except Exception:
                    # Skip one frame OCR cannot read rather than lose the
                    # whole pass; repeated failures stop the read below.
                    logger.warning("Could not read the overlay in frame %s of %s", index - 1, video_path.name,
                                   exc_info=True)
            else:
                logger.warning("Skipped undecodable frame %s in %s", index - 1, video_path.name)
            if row is None:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                    stopped = (f"{failures} sampled frames in a row could not be decoded or read "
                               f"at {(index - 1) / fps:.1f} s")
                    break
                continue
            failures = 0
            row["t_s"] = round((index - 1) / fps, 3)
            samples.append(row)
            if progress:
                progress(min(100.0, 100.0 * index / frame_count), f"Read live score at {row['t_s']:.1f}s")
    finally:
        cap.release()
    if not samples:
        raise RuntimeError(f"Could not read any frame of {video_path.name}.")
    duration = frame_count / fps
    if stopped:
        # Events after this point are missing, so callers must not treat the
        # proposals as covering the whole match.
        logger.warning("Stopped reading the live overlay of %s early: %s", video_path.name, stopped)
    return {"source": "live_overlay", "sample_fps": sample_fps, "layout": layout.as_dict(), "samples": samples,
            "complete": stopped is None, "stopped_early": stopped,
            "read_until_s": samples[-1]["t_s"], "duration_s": round(duration, 3)}


# A reading counts once this many consecutive readable samples agree on it.
MIN_STABLE_SAMPLES = 2


def stable_steps(samples: list[dict], key: str, min_samples: int = MIN_STABLE_SAMPLES) -> list[tuple[float, int]]:
    """(time, value) each time the stable value of one field changes.

    A value becomes stable once ``min_samples`` consecutive readable samples
    show it; unreadable samples are skipped without breaking a run. The time
    is that of the first sample in the run. The first stable value is
    included as the starting point.
    """
    steps: list[tuple[float, int]] = []
    run_value: int | None = None
    run_start = 0.0
    run_length = 0
    for sample in samples:
        value = sample.get(key)
        if value is None:
            continue
        if value == run_value:
            run_length += 1
        else:
            run_value, run_start, run_length = int(value), float(sample["t_s"]), 1
        if run_length == min_samples and (not steps or steps[-1][1] != value):
            steps.append((run_start, int(value)))
    return steps


AUTO, TRANSITION, TELEOP, ENDED = "AUTO", "TRANSITION", "TELEOP", "ENDED"
# The overlay clock counts 2:30 to 2:00 in AUTO, shows the TRANSITION countdown
# (8 s or less), then counts 2:00 to 0:00 in TELEOP.
_AUTO_END_CLOCK_S = 120
_TRANSITION_MAX_CLOCK_S = 8
# A stable clock at or above this after TELEOP or the end of a match is a new
# match starting (its AUTO counts down from 2:30).
_NEW_MATCH_MIN_CLOCK_S = 140


def match_phases(samples: list[dict], min_samples: int = MIN_STABLE_SAMPLES) -> list[tuple[float, str]]:
    """(time, phase) each time the match phase changes, from the stable clock.

    A clip may open on the previous match (its TELEOP or its 0:00 clock) or
    run into the next one, so a clock that jumps back up to AUTO time after
    TELEOP or the end starts a new AUTO.
    """
    phases: list[tuple[float, str]] = []
    phase: str | None = None
    previous: int | None = None
    for t_s, clock in stable_steps(samples, "clock_s", min_samples):
        if phase is None:
            # A clip opening on 0:01-0:08 may be in the TRANSITION countdown
            # or the last seconds of TELEOP; it is taken as TRANSITION until
            # the clock shows which (2:00 next, or 0:00).
            new = (AUTO if clock > _AUTO_END_CLOCK_S else ENDED if clock == 0
                   else TRANSITION if clock <= _TRANSITION_MAX_CLOCK_S else TELEOP)
        elif phase in (TELEOP, ENDED, TRANSITION) and clock >= _NEW_MATCH_MIN_CLOCK_S:
            # The countdown runs into 2:00, so 2:20 or more after it is the
            # next match; an opening "TRANSITION" before it was TELEOP.
            if phase == TRANSITION and len(phases) == 1:
                phases[0] = (phases[0][0], TELEOP)
            new = AUTO
        elif (phase == AUTO and clock <= _TRANSITION_MAX_CLOCK_S and previous is not None
              and abs(previous - _AUTO_END_CLOCK_S) <= 2):
            new = TRANSITION
        elif phase == AUTO and clock == 0:
            # Cut from mid-AUTO straight to the 0:00 post-match overlay (the
            # TRANSITION countdown only follows a clock at 2:00).
            new = ENDED
        elif phase == AUTO and _TRANSITION_MAX_CLOCK_S < clock < _AUTO_END_CLOCK_S:
            # The TRANSITION countdown was not read (or the stream cut it):
            # a clock below 2:00 that is not a countdown is TELEOP.
            new = TELEOP
        elif phase == TRANSITION and clock > _TRANSITION_MAX_CLOCK_S:
            new = TELEOP
        elif phase == TRANSITION and clock == 0:
            # The countdown always runs into 2:00, never 0:00 (checked on
            # every measured match), so an opening "TRANSITION" that reaches
            # 0:00 was the end of TELEOP; one that followed AUTO is a cut to
            # the post-match overlay.
            if len(phases) == 1:
                phases[0] = (phases[0][0], TELEOP)
            new = ENDED
        elif phase == TELEOP and clock == 0:
            new = ENDED
        else:
            new = phase
        if new != phase:
            phases.append((t_s, new))
            phase = new
        previous = clock
    return phases


def phase_at(phases: list[tuple[float, str]], t_s: float) -> str | None:
    current = None
    for start, phase in phases:
        if start <= t_s:
            current = phase
        else:
            break
    return current


# A score step within this many seconds of a TIP count step of the same
# alliance is that TIP being entered.
_TIP_PAIRING_S = 3.0
TIP_POINTS = 20
MINOR_FOUL_POINTS = 5
MAJOR_FOUL_POINTS = 20


FOUL_POINTS = {"minor_foul": MINOR_FOUL_POINTS, "major_foul": MAJOR_FOUL_POINTS}
# A correction is matched against at most this many of the latest open rises.
_CORRECTION_LOOKBACK = 6


def _apply_correction(drop: int, open_rises: list[dict], events: list[dict], log: list[dict],
                      alliance: str, t_s: float) -> int:
    """Undo unexplained score rises that a later score drop corrects.

    Scorekeepers fix entries by lowering the score (three TIPS entered, then
    two taken back). First look for recent rises that add up to the drop
    exactly, preferring the latest; cancel those. Otherwise take the drop
    off the latest rises, newest first, shrinking the last one if only part
    of it was corrected. Returns the part of the drop no rise accounts for.
    """
    recent = open_rises[-_CORRECTION_LOOKBACK:]
    for size in range(1, len(recent) + 1):
        for chosen in combinations(range(len(recent) - 1, -1, -1), size):
            if sum(recent[i]["points"] for i in chosen) == drop:
                for i in chosen:
                    rise = recent[i]
                    open_rises.remove(rise)
                    events.remove(rise)
                    log.append({"alliance": alliance, "t_s": t_s, "cancelled_rise_t_s": rise["t_s"],
                                "points": rise["points"]})
                return 0
    remaining = drop
    while remaining > 0 and open_rises:
        rise = open_rises[-1]
        if rise["points"] <= remaining:
            remaining -= rise["points"]
            open_rises.pop()
            events.remove(rise)
            log.append({"alliance": alliance, "t_s": t_s, "cancelled_rise_t_s": rise["t_s"], "points": rise["points"]})
        else:
            rise["points"] -= remaining
            rise["note"] += f" (reduced by {remaining} by a later correction)"
            if rise["event_type"] in FOUL_POINTS and rise["points"] != FOUL_POINTS[rise["event_type"]]:
                # A foul credit is only ever 5 or 20; what is left is not one.
                rise["event_type"] = "unclassified"
            log.append({"alliance": alliance, "t_s": t_s, "reduced_rise_t_s": rise["t_s"], "points": remaining})
            remaining = 0
    return remaining


def propose_events(samples: list[dict], min_samples: int = MIN_STABLE_SAMPLES) -> dict:
    """Turn overlay readings into flagged candidate scoring events.

    - Each rise of an alliance's stable TIP count is a HIVE TIP (20 points),
      AUTO if it is entered before TELEOP starts.
    - A score rise not explained by a TIP during TELEOP is a foul credit
      when it is 5 or 20 points, otherwise unclassified.
    - Score rises at the end of AUTO or after the match are period-end
      assessments (LEAVE, PARK, end-state elements) and are proposed as
      unclassified with their points, for the reviewer to split by type.
    - A score drop cancels the latest unexplained rise of the same size
      (a scorekeeper correction); otherwise it is proposed as a negative
      unclassified adjustment.

    Returns the events plus the stable steps they came from.
    """
    phases = match_phases(samples, min_samples)
    events: list[dict] = []
    detail: dict = {"phases": phases, "corrections": []}
    # A clip can start on the previous match's final score or run into the
    # next match. The match is the first AUTO that reaches TELEOP (else the
    # last AUTO seen); only readings from its start to the next AUTO count.
    autos = [index for index, (_, phase) in enumerate(phases) if phase == AUTO]
    played = [index for index in autos
              if any(phase == TELEOP for _, phase in phases[index + 1:next((a for a in autos if a > index), len(phases))])]
    chosen = played[0] if played else (autos[-1] if autos else None)
    in_match = samples
    detail["match_start_s"] = detail["match_end_s"] = None
    if chosen is not None:
        start = phases[chosen][0]
        end = next((phases[a][0] for a in autos if a > chosen), None)
        detail["match_start_s"], detail["match_end_s"] = start, end
        in_match = [s for s in samples if float(s["t_s"]) >= start and (end is None or float(s["t_s"]) < end)]
    for alliance in ("red", "blue"):
        tips = stable_steps(in_match, f"{alliance}_tips", min_samples)
        scores = stable_steps(in_match, f"{alliance}_score", min_samples)
        detail[f"{alliance}_tip_steps"] = tips
        detail[f"{alliance}_score_steps"] = scores
        tip_times: list[float] = []
        for (_, before), (t_s, after) in zip(tips, tips[1:]):
            if after > before:
                tip_times.extend([t_s] * (after - before))
            else:
                # A lower count is a correction: drop the latest TIPs.
                for _ in range(min(before - after, len(tip_times))):
                    removed = tip_times.pop()
                    detail["corrections"].append({"alliance": alliance, "t_s": t_s, "removed_tip_t_s": removed})
        unpaired = list(tip_times)
        for t_s in tip_times:
            phase = phase_at(phases, t_s)
            events.append({"t_s": t_s, "points": TIP_POINTS, "scoring_alliance": alliance,
                           "event_type": "hive_tip_auto" if phase in (None, AUTO, TRANSITION) else "hive_tip_teleop",
                           "source": "live_overlay", "status": "flagged", "confidence": "medium",
                           "note": "HIVE TIP count rose on the live overlay"})
        open_rises: list[dict] = []
        for step, ((previous_t, before), (t_s, after)) in enumerate(zip(scores, scores[1:]), start=1):
            delta = after - before
            if delta > 0:
                # The TIP count and the score are entered together, but either
                # can become readable later. A TIP pairs with this rise if it
                # is close in time or falls while the score showed this value
                # (between the previous and the next score change).
                next_t = scores[step + 1][0] if step + 1 < len(scores) else math.inf
                explained = 0
                while delta - explained >= TIP_POINTS:
                    candidates = [tip for tip in unpaired
                                  if abs(tip - t_s) <= _TIP_PAIRING_S or previous_t < tip < next_t]
                    match = min(candidates, key=lambda tip: abs(tip - t_s), default=None)
                    if match is None:
                        break
                    unpaired.remove(match)
                    explained += TIP_POINTS
                rest = delta - explained
                if rest <= 0:
                    continue
                phase = phase_at(phases, t_s)
                if phase == TELEOP and rest == MINOR_FOUL_POINTS:
                    event_type, note = "minor_foul", "Score rose 5 during TELEOP without a TIP: likely a MINOR FOUL credit"
                elif phase == TELEOP and rest == MAJOR_FOUL_POINTS:
                    event_type, note = "major_foul", "Score rose 20 during TELEOP without a TIP: likely a MAJOR FOUL credit"
                elif phase in (AUTO, TRANSITION) and rest == MAJOR_FOUL_POINTS:
                    # LEAVE and AUTO PARK give an alliance at most 16 points,
                    # so 20 without a TIP cannot be those.
                    event_type, note = "major_foul", "Score rose 20 in AUTO without a TIP: likely a MAJOR FOUL credit"
                elif phase in (AUTO, TRANSITION):
                    event_type, note = "unclassified", "Score rose at the end of AUTO: LEAVE/PARK or a foul credit"
                elif phase == ENDED:
                    event_type, note = "unclassified", "Score rose after the match: end-of-match assessment (PARK, CELL, FLOWER, GARDEN) or a foul credit"
                else:
                    event_type, note = "unclassified", "Score rose without a TIP"
                event = {"t_s": t_s, "points": rest, "scoring_alliance": alliance, "event_type": event_type,
                         "source": "live_overlay", "status": "flagged", "confidence": "low", "note": note}
                events.append(event)
                open_rises.append(event)
            elif delta < 0:
                remaining = _apply_correction(-delta, open_rises, events, detail["corrections"], alliance, t_s)
                if remaining:
                    events.append({"t_s": t_s, "points": -remaining, "scoring_alliance": alliance,
                                   "event_type": "unclassified", "source": "live_overlay", "status": "flagged",
                                   "confidence": "low", "note": "Score dropped on the live overlay: a correction"})
        detail[f"{alliance}_unpaired_tip_times"] = unpaired
    events.sort(key=lambda event: (event["t_s"], event["scoring_alliance"]))
    return {"events": events, **detail}


def summarize(proposal: dict) -> dict:
    """Per-alliance totals of the proposed events, for comparison with official results."""
    summary = {}
    for alliance in ("red", "blue"):
        mine = [event for event in proposal["events"] if event["scoring_alliance"] == alliance]
        scores = proposal.get(f"{alliance}_score_steps") or []
        summary[alliance] = {
            "auto_tips": sum(1 for event in mine if event["event_type"] == "hive_tip_auto"),
            "teleop_tips": sum(1 for event in mine if event["event_type"] == "hive_tip_teleop"),
            "foul_credit": sum(event["points"] for event in mine if event["event_type"] in {"minor_foul", "major_foul"}),
            "event_points": sum(event["points"] for event in mine),
            "last_stable_score": scores[-1][1] if scores else None,
        }
    return summary
