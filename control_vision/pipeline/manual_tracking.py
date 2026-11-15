from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from control_vision.image_files import write_image
from control_vision.pipeline.homography import matrix_from_corners, project_point

ALLIANCES = ("red", "blue")
MIN_SAMPLE_FPS, MAX_SAMPLE_FPS = 5.0, 10.0
# Rows below this confidence are never emitted. Every uncertain state
# (searching, ambiguous, occluded) is capped below it.
EMIT_THRESHOLD = 0.3
MAX_CONFIDENCE = 0.9
OCCLUSION_CAP = 0.28
SEARCH_CONFIDENCE = 0.1
TRUST_DECAY = 0.75
TRUST_RECOVERY = 0.15
VERIFY_SIMILARITY = 0.45
EMIT_SIMILARITY = 0.65
PROBATION_FRAMES = 2
REACQUIRE_SIMILARITY = 0.75
REACQUIRE_COLOUR_RATIO = 0.75
REACQUIRE_MARGIN = 0.1
APPEARANCE_MARGIN = 0.05
AMBIGUITY_WIDTHS = 1.25
MIN_COLOUR_RATIO = 0.35
MIN_SEED_COLOUR_PX = 30
BLOCKING_SEARCH_FRAMES = 16
MAX_COLOURLESS_FRAMES = 2
MAX_READ_FAILURES = 5
PHOTO_MARGIN = 0.2
MIN_PHOTO_SIDE = 200

Box = tuple[float, float, float, float]


class _SeedOutsideFrame(ValueError):
    """A clicked seed point does not lie inside the decoded video frame."""


def _alliance_mask(hsv: np.ndarray, alliance: str) -> np.ndarray:
    if alliance == "red":
        low = cv2.inRange(hsv, np.array((0, 75, 45)), np.array((12, 255, 255)))
        high = cv2.inRange(hsv, np.array((168, 75, 45)), np.array((180, 255, 255)))
        return cv2.bitwise_or(low, high)
    return cv2.inRange(hsv, np.array((88, 65, 35)), np.array((137, 255, 255)))


def _box(anchor: tuple[float, float], width: float, height: float) -> Box:
    """Robot box whose bottom-centre is the ground-contact anchor."""
    x, y = anchor
    return (x - width / 2, y - height, x + width / 2, y + 3.0)


def _expand(box: Box, fx: float, fy: float) -> Box:
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - w * fx, box[1] - h * fy, box[2] + w * fx, box[3] + h * fy)


def _clip(box: Box, shape: tuple[int, ...], minimum: int = 8) -> tuple[int, int, int, int] | None:
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x1, y1 = min(shape[1], int(round(box[2]))), min(shape[0], int(round(box[3])))
    if x1 - x0 < minimum or y1 - y0 < minimum:
        return None
    return x0, y0, x1, y1


def _inside(point: tuple[float, float], box: Box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _components(hsv: np.ndarray, alliance: str, window: Box):
    clipped = _clip(window, hsv.shape)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    mask = _alliance_mask(hsv[y0:y1, x0:x1], alliance)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return x0, y0, mask


def _field_size(inverse: np.ndarray, anchor: tuple[float, float], patch_width: int,
                patch_height: int) -> tuple[float, float]:
    """Pixel size of an 18-inch robot standing at the anchor, from the field calibration."""
    x_ft, y_ft = project_point(np.linalg.inv(inverse), *anchor)
    left = project_point(inverse, x_ft - 0.75, y_ft)
    right = project_point(inverse, x_ft + 0.75, y_ft)
    width = math.hypot(right[0] - left[0], right[1] - left[1])
    if not math.isfinite(width):
        return float(patch_width), float(patch_height)
    width = float(np.clip(width, 0.4 * patch_width, 2.0 * patch_width))
    return width, float(np.clip(0.8 * width, 0.4 * patch_height, 2.0 * patch_height))


def _seed_box(hsv: np.ndarray, alliance: str, anchor: tuple[float, float],
              patch_width: int, patch_height: int,
              default: tuple[float, float] | None = None) -> tuple[float, float, bool]:
    """Estimate the robot box from alliance-colour blobs around the click.

    Returns (width, height, colour found). The click itself stays the
    ground-contact anchor; colour only sizes the box (bumpers, side panels or a
    flag on top all count).
    """
    ax, ay = anchor
    reach = (ax - 0.75 * patch_width, ay - 1.2 * patch_height, ax + 0.75 * patch_width, ay + 6)
    found = _components(hsv, alliance, _expand(reach, 1.0, 0.3))
    union: list[float] | None = None
    if found is not None:
        x0, y0, mask = found
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            bx, by, bw, bh, area = (int(v) for v in stats[index])
            if area < 24 or bw < 8 or bh < 4 or bw > 2.5 * patch_width or bh > 2.5 * patch_height:
                continue
            rect = (x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh)
            if not _inside(((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2), reach):
                continue
            union = list(rect) if union is None else [min(union[0], rect[0]), min(union[1], rect[1]),
                                                       max(union[2], rect[2]), max(union[3], rect[3])]
    if union is None:
        if default is not None:
            return default[0], default[1], False
        return float(patch_width), float(patch_height), False
    width = float(np.clip(union[2] - union[0], 0.5 * patch_width, 3.0 * patch_width))
    height = float(np.clip(max(width * patch_height / patch_width, ay - union[1]), 0.5 * patch_height, 2.5 * patch_height))
    return width, height, True


def _colour(hsv: np.ndarray, alliance: str, box: Box, seed_px: float):
    """Alliance-colour pixels in the robot box and whether a second robot-sized
    blob of the same colour sits in the surrounding window."""
    found = _components(hsv, alliance, _expand(box, 1.0, 0.5))
    if found is None:
        return 0.0, False
    x0, y0, mask = found
    inner = _clip(_expand(box, 0.15, 0.15), hsv.shape, minimum=1)
    pixels = 0.0
    if inner is not None:
        ix0, iy0, ix1, iy1 = inner
        sub = mask[max(0, iy0 - y0):max(0, iy1 - y0), max(0, ix0 - x0):max(0, ix1 - x0)]
        pixels = float(np.count_nonzero(sub))
    other = False
    if seed_px >= MIN_SEED_COLOUR_PX:
        w, h = box[2] - box[0], box[3] - box[1]
        own = _expand(box, 0.25, 0.25)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            bw, bh, area = int(stats[index][2]), int(stats[index][3]), int(stats[index][4])
            if area < 0.5 * seed_px or bw > 3.2 * w or bh > 1.6 * h:
                continue
            centre = (x0 + float(centroids[index][0]), y0 + float(centroids[index][1]))
            # A robot-sized blob beside the robot, or the robot's own colour
            # merged with a second robot's: far more colour than at the seed,
            # or a bumper-thick blob much wider than this robot. Thin lines
            # such as field tape are neither.
            beside = not _inside(centre, own) and bw <= 1.6 * w
            merged = area >= 1.3 * seed_px or (bw > 1.25 * w and bh >= 0.15 * h)
            if beside or merged:
                other = True
                break
    return pixels, other


def _resized(template: np.ndarray, width: int, height: int) -> np.ndarray:
    th, tw = template.shape[:2]
    if abs(tw - width) <= max(2, 0.03 * width) and abs(th - height) <= max(2, 0.03 * height):
        return template
    return cv2.resize(template, (max(4, width), max(4, height)), interpolation=cv2.INTER_AREA)


def _match(smooth: np.ndarray, template: np.ndarray | None, box: Box, pad_x: float, pad_y: float):
    """Best normalised template match near a box.

    Returns (best score, second-best score away from the best, best shift,
    second shift), shifts being relative to the box's top-left corner."""
    if template is None:
        return None
    bx0, by0 = int(round(box[0])), int(round(box[1]))
    width, height = int(round(box[2] - box[0])), int(round(box[3] - box[1]))
    if width < 8 or height < 8:
        return None
    tmpl = _resized(template, width, height)
    th, tw = tmpl.shape[:2]
    rx0, ry0 = max(0, int(round(bx0 - pad_x))), max(0, int(round(by0 - pad_y)))
    rx1 = min(smooth.shape[1], int(round(bx0 + tw + pad_x)))
    ry1 = min(smooth.shape[0], int(round(by0 + th + pad_y)))
    if rx1 - rx0 < tw or ry1 - ry0 < th:
        return None
    region = smooth[ry0:ry1, rx0:rx1]
    result = np.nan_to_num(cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED), nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, best, _, loc = cv2.minMaxLoc(result)
    patch = region[loc[1]:loc[1] + th, loc[0]:loc[0] + tw]
    if float(patch.std()) < 2.0:
        best = 0.0
    sx, sy = max(1, tw // 2), max(1, th // 2)
    masked = result.copy()
    masked[max(0, loc[1] - sy):loc[1] + sy + 1, max(0, loc[0] - sx):loc[0] + sx + 1] = -1.0
    _, second, _, loc2 = cv2.minMaxLoc(masked)
    return (float(best), float(second), (float(rx0 + loc[0] - bx0), float(ry0 + loc[1] - by0)),
            (float(rx0 + loc2[0] - bx0), float(ry0 + loc2[1] - by0)))


def _template(smooth: np.ndarray, box: Box) -> np.ndarray | None:
    clipped = _clip(box, smooth.shape)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    patch = smooth[y0:y1, x0:x1].copy()
    return patch if float(patch.std()) >= 2.0 else None


def _features(gray: np.ndarray, box: Box, smooth: np.ndarray | None = None,
              template: np.ndarray | None = None) -> np.ndarray | None:
    """Corners inside the robot box. With a template, only pixels that still
    look like the robot may hold a corner, so an occluder or a static field
    edge inside the box cannot steer the flow."""
    clipped = _clip(_expand(box, -0.08, -0.05), gray.shape, minimum=16)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    full = _clip(box, gray.shape)
    if smooth is not None and template is not None and full is not None:
        fx0, fy0, fx1, fy1 = full
        bx0, by0 = int(round(box[0])), int(round(box[1]))
        tmpl = _resized(template, int(round(box[2] - box[0])), int(round(box[3] - box[1])))
        part = tmpl[fy0 - by0:fy1 - by0, fx0 - bx0:fx1 - bx0]
        patch = smooth[fy0:fy1, fx0:fx1]
        if part.shape == patch.shape:
            differs = np.abs(patch - part).mean(axis=2) > 30
            agree = np.where(differs, 0, 255).astype(np.uint8)
            mask[fy0:fy1, fx0:fx1] = cv2.bitwise_and(mask[fy0:fy1, fx0:fx1], agree)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.012,
                                     minDistance=4, blockSize=7, mask=mask)
    return points if points is not None and len(points) >= 4 else None


def _confidence(flow_quality: float, colour_ratio: float, similarity: float) -> float:
    confidence = min(MAX_CONFIDENCE, 0.5 * flow_quality + 0.25 * min(1.0, colour_ratio)
                     + 0.25 * max(0.0, similarity))
    if colour_ratio < MIN_COLOUR_RATIO or similarity < EMIT_SIMILARITY:
        # Flow without visible alliance colour, or with the robot only partly
        # matching its template, keeps continuity through a short occlusion,
        # but such points never pass as reviewed positions.
        confidence = min(confidence, OCCLUSION_CAP)
    return confidence


@dataclass
class _Track:
    rid: int
    alliance: str
    anchor: tuple[float, float]
    width: float
    height: float
    seed_width: float
    seed_height: float
    template: np.ndarray | None
    colour_px: float
    features: np.ndarray | None
    velocity: tuple[float, float] = (0.0, 0.0)
    trust: float = 1.0
    searching: bool = False
    frames_lost: int = 0
    colourless: int = 0
    probation: int = 0

    def box(self) -> Box:
        return _box(self.anchor, self.width, self.height)

    def predicted(self) -> tuple[float, float]:
        return self.anchor[0] + self.velocity[0], self.anchor[1] + self.velocity[1]


@dataclass
class _Proposal:
    anchor: tuple[float, float]
    width: float
    height: float
    flow_quality: float
    similarity: float
    colour_ratio: float
    other_blob: bool
    reacquired: bool


def _flow(prev_gray: np.ndarray, gray: np.ndarray, track: _Track):
    old_points = track.features
    if old_points is None or len(old_points) < 4:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    # OpenCV stubs type nextPts as UMat; passing None asks OpenCV to allocate it.
    new_points, status, errors = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        prev_gray, gray, old_points, None, winSize=(21, 21), maxLevel=3, criteria=criteria)
    if new_points is None or status is None:
        return None
    # Forward-backward check: points hidden by an occluder or lost in a blur do
    # not track back to where they started, so they cannot vote for the motion.
    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        gray, prev_gray, new_points, None, winSize=(21, 21), maxLevel=3, criteria=criteria)
    if back_points is None or back_status is None:
        return None
    round_trip = np.linalg.norm(back_points.reshape(-1, 2) - old_points.reshape(-1, 2), axis=1)
    good = (status.ravel() == 1) & (back_status.ravel() == 1) & (round_trip < 1.0)
    if errors is not None:
        good &= errors.ravel() < 35
    src, dst = old_points.reshape(-1, 2)[good], new_points.reshape(-1, 2)[good]
    if len(dst) < 4:
        return None
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if affine is None or inliers is None or int(inliers.sum()) < 4:
        return None
    ax, ay = track.anchor
    nx = float(affine[0, 0] * ax + affine[0, 1] * ay + affine[0, 2])
    ny = float(affine[1, 0] * ax + affine[1, 1] * ay + affine[1, 2])
    scale = float(math.hypot(affine[0, 0], affine[1, 0]))
    if not 0.8 <= scale <= 1.25 or math.hypot(nx - ax, ny - ay) > track.width:
        return None
    inlier_fraction = float(np.mean(inliers))
    quality = min(0.78, 0.35 + 0.55 * inlier_fraction * min(1.0, int(inliers.sum()) / 12))
    return (nx, ny), scale, quality


def _flow_proposal(prev_gray, gray, smooth, hsv, track: _Track) -> _Proposal | None:
    flowed = _flow(prev_gray, gray, track)
    if flowed is None:
        return None
    anchor, scale, quality = flowed
    width, height = track.width, track.height
    if abs(scale - 1.0) > 0.03:
        width = float(np.clip(width * scale, 0.6 * track.seed_width, 1.6 * track.seed_width))
        height = float(np.clip(height * scale, 0.6 * track.seed_height, 1.6 * track.seed_height))
    box = _box(anchor, width, height)
    # The template search corrects flow that a partial occluder or a static
    # edge in the box has pulled off the robot.
    pad = max(4.0, 0.2 * width)
    matched = _match(smooth, track.template, box, pad, pad)
    if matched is None:
        return None
    similarity, _, shift, _ = matched
    if similarity >= VERIFY_SIMILARITY:
        anchor = (anchor[0] + shift[0], anchor[1] + shift[1])
        box = _box(anchor, width, height)
    pixels, other = _colour(hsv, track.alliance, box, track.colour_px)
    ratio = pixels / track.colour_px if track.colour_px >= MIN_SEED_COLOUR_PX else 0.0
    return _Proposal(anchor, width, height, quality, similarity, ratio, other, False)


def _reacquire(smooth, half, hsv, track: _Track, tracks: list[_Track]) -> _Proposal | None:
    """Template search near the predicted position; accept only one clear match."""
    if track.template is None or (track.frames_lost > 8 and track.frames_lost % 2):
        return None
    centre = track.predicted()
    radius = track.width * min(2.0, 0.75 + 0.25 * track.frames_lost)
    box = _box(centre, track.width, track.height)
    # Coarse search at half resolution keeps long searches cheap; the best
    # match is then confirmed at full resolution.
    coarse = _match(half, track.template, (box[0] / 2, box[1] / 2, box[2] / 2, box[3] / 2),
                    radius / 2, 0.375 * radius)
    if coarse is None:
        return None
    best, second, shift, second_shift = coarse
    if best < REACQUIRE_SIMILARITY - 0.1:
        return None
    candidates = [shift] + ([second_shift] if second > REACQUIRE_SIMILARITY - 0.3 else [])
    scored = []
    for coarse_shift in candidates:
        start = (centre[0] + 2 * coarse_shift[0], centre[1] + 2 * coarse_shift[1])
        fine = max(4.0, 0.2 * track.width)
        refined = _match(smooth, track.template, _box(start, track.width, track.height), fine, fine)
        if refined is not None:
            scored.append((refined[0], (start[0] + refined[2][0], start[1] + refined[2][1])))
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])
    best, anchor = scored[0]
    # A second location that matches almost as well means the template cannot
    # tell which robot is which; stay suspended instead of guessing.
    if best < REACQUIRE_SIMILARITY or (len(scored) > 1 and best - scored[1][0] < REACQUIRE_MARGIN):
        return None
    box = _box(anchor, track.width, track.height)
    pad = max(3.0, 0.06 * track.width)
    for other in tracks:
        if other is track or other.alliance != track.alliance:
            continue
        rival = _match(smooth, other.template, box, pad, pad)
        if rival is not None and rival[0] + APPEARANCE_MARGIN > best:
            return None
    pixels, blob = _colour(hsv, track.alliance, box, track.colour_px)
    ratio = pixels / track.colour_px if track.colour_px >= MIN_SEED_COLOUR_PX else 0.0
    if ratio < REACQUIRE_COLOUR_RATIO:
        return None
    quality = min(0.75, 0.35 + (best - REACQUIRE_SIMILARITY))
    return _Proposal(anchor, track.width, track.height, quality, best, ratio, blob, True)


def _ambiguous_pairs(tracks: list[_Track], proposals: dict[int, _Proposal | None]) -> set[int]:
    """Same-alliance tracks closer than about one robot width cannot be told
    apart reliably, so both are suspended rather than guessing."""
    positions: dict[int, tuple[float, float]] = {}
    for track in tracks:
        proposal = proposals.get(track.rid)
        if proposal is not None:
            positions[track.rid] = proposal.anchor
        elif track.frames_lost <= BLOCKING_SEARCH_FRAMES:
            positions[track.rid] = track.predicted()
    flagged: set[int] = set()
    for i, first in enumerate(tracks):
        for second in tracks[i + 1:]:
            if first.alliance != second.alliance or first.rid not in positions or second.rid not in positions:
                continue
            p, q = positions[first.rid], positions[second.rid]
            if math.hypot(p[0] - q[0], p[1] - q[1]) < AMBIGUITY_WIDTHS * max(first.width, second.width):
                flagged.update((first.rid, second.rid))
    return flagged


def _photo_bounds(shape: tuple[int, ...], box: Box) -> tuple[int, int, int, int] | None:
    return _clip(_expand(box, PHOTO_MARGIN, PHOTO_MARGIN), shape, minimum=1)


def _offer_photo(best_photos: dict[int, tuple[float, float, float, np.ndarray]], rid: int,
                 frame: np.ndarray, box: Box, confidence: float, t_s: float) -> None:
    """Keep the frame with the largest robot box weighted by confidence."""
    bounds = _photo_bounds(frame.shape, box)
    if bounds is None:
        return
    score = (box[2] - box[0]) * (box[3] - box[1]) * confidence
    if rid not in best_photos or score > best_photos[rid][0]:
        x0, y0, x1, y1 = bounds
        best_photos[rid] = (score, confidence, t_s, frame[y0:y1, x0:x1].copy())


def _validate_inputs(seed_points, alliances, sample_fps) -> tuple[list[tuple[int, float, float]], dict[int, str]]:
    if not isinstance(sample_fps, (int, float)) or not math.isfinite(sample_fps) \
            or not MIN_SAMPLE_FPS <= sample_fps <= MAX_SAMPLE_FPS:
        raise ValueError(f"Sample frame rate must be between {MIN_SAMPLE_FPS:g} and {MAX_SAMPLE_FPS:g} frames per second.")
    if not seed_points:
        raise ValueError("Seed at least one robot before tracking.")
    seeds: list[tuple[int, float, float]] = []
    for item in seed_points:
        rid, x, y = int(item[0]), float(item[1]), float(item[2])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Robot {rid} seed point is not a valid pixel position.")
        if any(rid == seed[0] for seed in seeds):
            raise ValueError(f"Robot {rid} is seeded more than once.")
        seeds.append((rid, x, y))
    normalised: dict[int, str] = {}
    for rid, _, _ in seeds:
        alliance = alliances.get(rid)
        if alliance is None:
            raise ValueError(f"Robot {rid} has no alliance. Set it to red or blue.")
        value = str(alliance).strip().lower()
        if value not in ALLIANCES:
            raise ValueError(f"Robot {rid} alliance {alliance!r} is not red or blue.")
        normalised[rid] = value
    return seeds, normalised


def track_seeded_video(video_path: Path, corners: list[list[float]],
                       seed_points: list[tuple[int, int, int]], alliances: dict[int, str],
                       sample_fps: float, cancel: Callable[[], bool],
                       progress: Callable[[str, float, str], None], logger: logging.Logger,
                       photos_dir: Path | None = None,
                       patch_width: int = 112, patch_height: int = 88):
    """Track clicked robots with Lucas–Kanade flow checked against each robot's
    template and alliance colour. No model file is used.

    This is a deliberately local preview for a human-initialised full-field video.
    The clicked ground-contact point is the projected anchor and moves with the
    tracked displacement. A wrong identity is worse than a gap: when two
    same-alliance robots are too close to tell apart, when the template no
    longer matches, or when the robot is lost, the track is suspended and emits
    nothing until one clear template match brings it back.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video file {video_path.name}. It may be missing, corrupt, or in an unsupported format.")
    try:
        seeds, alliance_of = _validate_inputs(seed_points, alliances, sample_fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
            raise RuntimeError(f"Could not read frame rate or frame count from {video_path.name}.")
        stride = max(1, round(fps / sample_fps))
        sampled_total = max(1, math.ceil(total_frames / stride))
        matrix = matrix_from_corners(corners)
    except Exception:
        cap.release()
        raise
    tracks: list[_Track] = []
    rows: list[tuple[int, int, float, float, float, float]] = []
    best_photos: dict[int, tuple[float, float, float, np.ndarray]] = {}
    sampled = 0
    decoded = 0
    failures = 0
    frame_index = 0
    frame_shape: tuple[int, ...] | None = None
    prev_gray: np.ndarray | None = None
    try:
        while frame_index < total_frames:
            if frame_index % stride:
                if not cap.grab():
                    failures += 1
                    logger.warning("Could not grab frame %d of %s; skipped", frame_index, video_path.name)
                    if failures >= MAX_READ_FAILURES:
                        logger.warning("Stopped reading %s after %d consecutive unreadable frames at frame %d",
                                       video_path.name, failures, frame_index)
                        break
                frame_index += 1
                continue
            if cancel():
                raise InterruptedError("Manual tracking was cancelled.")
            t_s = frame_index / fps
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                logger.warning("Could not read frame %d (%.2fs) of %s; skipped", frame_index, t_s, video_path.name)
                if failures >= MAX_READ_FAILURES:
                    logger.warning("Stopped reading %s after %d consecutive unreadable frames at frame %d",
                                   video_path.name, failures, frame_index)
                    break
            else:
                failures = 0
                try:
                    if frame.ndim != 3 or frame.shape[2] != 3 or (frame_shape is not None and frame.shape != frame_shape):
                        raise ValueError(f"unexpected frame shape {frame.shape}")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    smooth = cv2.GaussianBlur(frame, (3, 3), 0).astype(np.float32)
                    if frame_shape is None:
                        tracks = _start_tracks(seeds, alliance_of, frame, gray, smooth, hsv,
                                               patch_width, patch_height, np.linalg.inv(matrix), logger)
                        frame_shape = frame.shape
                        emitted = [(track, _seed_confidence(track)) for track in tracks]
                    else:
                        assert prev_gray is not None
                        half = cv2.resize(smooth, (smooth.shape[1] // 2, smooth.shape[0] // 2),
                                          interpolation=cv2.INTER_AREA)
                        emitted = _step(tracks, prev_gray, gray, smooth, half, hsv, t_s, logger)
                    for track, confidence in emitted:
                        if confidence < EMIT_THRESHOLD:
                            continue
                        x_ft, y_ft = project_point(matrix, *track.anchor)
                        if not (0 <= x_ft <= 12 and 0 <= y_ft <= 12):
                            # Bad calibration or a lost track must not enter the field map.
                            continue
                        rows.append((track.rid, frame_index, t_s, x_ft, y_ft, confidence))
                        _offer_photo(best_photos, track.rid, frame, track.box(), confidence, t_s)
                    prev_gray = gray
                    decoded += 1
                except _SeedOutsideFrame:
                    raise
                except (cv2.error, ValueError) as exc:
                    logger.warning("Skipped frame %d (%.2fs) of %s: %s", frame_index, t_s, video_path.name, exc)
            sampled += 1
            progress("Seeded optical-flow tracking", min(100.0, 100.0 * sampled / sampled_total),
                     f"Processed {sampled:,} of {sampled_total:,} sampled frames at {t_s:.1f}s")
            frame_index += 1
    finally:
        cap.release()
    if decoded == 0:
        raise RuntimeError(f"Could not decode any frames from {video_path.name}.")
    photo_rows = _save_best_photos(best_photos, photos_dir, logger)
    logger.info("Manual tracker seeds: %s; track points: %d; photos: %d",
                {rid: (x, y) for rid, x, y in seeds}, len(rows), len(photo_rows))
    return rows, photo_rows, []


def _start_tracks(seeds, alliance_of, frame, gray, smooth, hsv, patch_width, patch_height,
                  inverse: np.ndarray, logger: logging.Logger) -> list[_Track]:
    height_px, width_px = frame.shape[:2]
    tracks: list[_Track] = []
    for rid, x, y in seeds:
        if not (0 <= x < width_px and 0 <= y < height_px):
            raise _SeedOutsideFrame(f"Robot {rid} seed point ({x:g}, {y:g}) is outside the {width_px}x{height_px} video frame.")
        alliance = alliance_of[rid]
        width, height, coloured = _seed_box(hsv, alliance, (x, y), patch_width, patch_height,
                                            _field_size(inverse, (x, y), patch_width, patch_height))
        box = _box((x, y), width, height)
        pixels, _ = _colour(hsv, alliance, box, 0.0)
        if pixels < max(MIN_SEED_COLOUR_PX, 0.02 * width * height) or not coloured:
            # Without the robot's own alliance colour there is nothing to tell
            # it from the field, so every point stays below the threshold.
            logger.warning("Robot %s seed has no %s alliance colour near (%.0f, %.0f); it will stay below the "
                           "confidence threshold", rid, alliance, x, y)
            pixels = 0.0
        template = _template(smooth, box)
        features = _features(gray, box)
        logger.info("Robot %s seeded at (%.0f, %.0f) with a %.0fx%.0f px box, %d features, %d %s pixels",
                    rid, x, y, width, height, 0 if features is None else len(features), pixels, alliance)
        tracks.append(_Track(rid, alliance, (x, y), width, height, width, height,
                             template, pixels, features))
    return tracks


def _seed_confidence(track: _Track) -> float:
    flow_quality = 0.78 if track.features is not None else 0.25
    ratio = 1.0 if track.colour_px >= MIN_SEED_COLOUR_PX else 0.0
    return _confidence(flow_quality, ratio, 1.0 if track.template is not None else 0.0)


def _step(tracks: list[_Track], prev_gray, gray, smooth, half, hsv, t_s: float,
          logger: logging.Logger) -> list[tuple[_Track, float]]:
    proposals: dict[int, _Proposal | None] = {}
    for track in tracks:
        if track.searching:
            proposals[track.rid] = _reacquire(smooth, half, hsv, track, tracks)
        else:
            proposals[track.rid] = _flow_proposal(prev_gray, gray, smooth, hsv, track)
    ambiguous = _ambiguous_pairs(tracks, proposals)
    emitted: list[tuple[_Track, float]] = []
    for track in tracks:
        proposal = proposals[track.rid]
        reason = None
        if proposal is None:
            reason = "lost"
        elif track.rid in ambiguous:
            reason = "another same-alliance robot is within a robot width"
        elif proposal.other_blob:
            reason = "another same-alliance robot-sized blob is in the search window"
        elif proposal.similarity < VERIFY_SIMILARITY:
            reason = "robot appearance no longer matches"
        elif proposal.colour_ratio < MIN_COLOUR_RATIO and track.colourless >= MAX_COLOURLESS_FRAMES:
            # Flow under a long occlusion follows whatever covers the robot.
            reason = "alliance colour hidden for too long"
        if reason is not None or proposal is None:
            if not track.searching:
                logger.info("Track %s suspended at %.2fs: %s", track.rid, t_s, reason)
            track.searching = True
            track.frames_lost += 1
            track.anchor = track.predicted()
            track.velocity = (track.velocity[0] * 0.95, track.velocity[1] * 0.95)
            track.trust *= TRUST_DECAY
            track.features = None
            track.colourless = 0
            emitted.append((track, min(SEARCH_CONFIDENCE, track.trust)))
            continue
        if proposal.reacquired:
            logger.info("Track %s reacquired at %.2fs (template score %.2f)", track.rid, t_s, proposal.similarity)
            track.velocity = (0.0, 0.0)
        else:
            dx, dy = proposal.anchor[0] - track.anchor[0], proposal.anchor[1] - track.anchor[1]
            track.velocity = (0.6 * track.velocity[0] + 0.4 * dx, 0.6 * track.velocity[1] + 0.4 * dy)
        track.searching = False
        track.frames_lost = 0
        track.colourless = track.colourless + 1 if proposal.colour_ratio < MIN_COLOUR_RATIO else 0
        track.anchor = proposal.anchor
        track.width, track.height = proposal.width, proposal.height
        confidence = _confidence(proposal.flow_quality, proposal.colour_ratio, proposal.similarity)
        if proposal.reacquired:
            track.probation = PROBATION_FRAMES
        elif track.probation and proposal.similarity >= EMIT_SIMILARITY:
            track.probation -= 1
        if track.probation:
            # One template match is not enough to resume: following frames must
            # confirm it with flow before the track is emitted again.
            confidence = min(confidence, OCCLUSION_CAP)
        if proposal.colour_ratio >= MIN_COLOUR_RATIO:
            track.trust = min(1.0, track.trust + TRUST_RECOVERY)
            if (not proposal.reacquired and proposal.similarity >= 0.75 and proposal.colour_ratio >= 0.8
                    and track.template is not None):
                patch = _template(smooth, track.box())
                if patch is not None:
                    th, tw = track.template.shape[:2]
                    patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
                    track.template = (0.85 * track.template + 0.15 * patch).astype(np.float32)
        track.features = _features(gray, track.box(), smooth, track.template)
        emitted.append((track, min(confidence, track.trust)))
    return emitted


def _save_best_photos(best_photos: dict[int, tuple[float, float, float, np.ndarray]],
                      photos_dir: Path | None, logger: logging.Logger) -> list[tuple[int, Path, float]]:
    photo_rows: list[tuple[int, Path, float]] = []
    if photos_dir is None:
        return photo_rows
    photos_dir.mkdir(parents=True, exist_ok=True)
    for rid, (_, _, t_s, crop) in best_photos.items():
        if crop.shape[0] < MIN_PHOTO_SIDE or crop.shape[1] < MIN_PHOTO_SIDE:
            logger.info("Robot %s preview photo skipped: best crop is %dx%d, below %dx%d",
                        rid, crop.shape[1], crop.shape[0], MIN_PHOTO_SIDE, MIN_PHOTO_SIDE)
            continue
        path = photos_dir / f"robot-{rid}.png"
        if write_image(path, crop):
            photo_rows.append((rid, path, t_s))
        else:
            logger.warning("Robot %s preview photo could not be written to %s", rid, path)
    return photo_rows
