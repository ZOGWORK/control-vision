from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from typing import Any

FIELD_SIZE_FT = 12.0
"""BIOBUZZ field is 12 x 12 ft; top-down origin at one field corner."""

FIELD_TOLERANCE_FT = 0.5
"""Homography slack: samples this far outside the field are clamped onto it."""

DEFAULT_DEBOUNCE_S = 0.5
DEFAULT_MAX_GAP_S = 1.0

Sample = tuple[float, float, float, float | None]
"""(t_s, x_ft, y_ft, conf) for one robot, top-down feet."""


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Even/odd ray casting; polygon points are top-down feet."""
    inside = False
    if len(polygon) < 3:
        return False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def is_roi_zone(zone: Mapping[str, Any]) -> bool:
    """Goal ROIs (types ending in ``_roi``) are camera regions, not field zones."""
    return str(zone.get("type", "")).endswith("_roi")


def field_zone_names(zones: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Field zones in config order, excluding ``_roi`` zones."""
    return [name for name, zone in zones.items() if not is_roi_zone(zone)]


def zone_at(x: float, y: float, zones: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Return the field zone containing (x, y), or None.

    ``_roi`` zones are skipped. Overlapping zones resolve deterministically to
    the first match in the mapping's iteration order (the zones_json order).
    """
    for name, zone in zones.items():
        if is_roi_zone(zone):
            continue
        if point_in_polygon(x, y, zone.get("points", [])):
            return name
    return None


def field_position(x: float, y: float, field_size_ft: float = FIELD_SIZE_FT,
                   tolerance_ft: float = FIELD_TOLERANCE_FT) -> tuple[float, float] | None:
    """Clamp a top-down position onto the field, or drop it.

    Positions within ``tolerance_ft`` of the field edge are clamped to the
    edge (homography slack). Anything further out, or non-finite, is a
    projection or identity error and returns None so the sample is dropped.
    """
    if not (isfinite(x) and isfinite(y)):
        return None
    low, high = -tolerance_ft, field_size_ft + tolerance_ft
    if not (low <= x <= high and low <= y <= high):
        return None
    return min(max(x, 0.0), field_size_ft), min(max(y, 0.0), field_size_ft)


def robot_samples(track_rows: Iterable[Sequence[Any]], field_size_ft: float = FIELD_SIZE_FT,
                  tolerance_ft: float = FIELD_TOLERANCE_FT) -> dict[int, list[Sample]]:
    """Group ``(robot_id, frame, t_s, x_ft, y_ft[, conf])`` rows per robot.

    Rows are sorted by (t_s, frame) so out-of-order input is handled, and
    off-field samples are clamped or dropped via :func:`field_position`.
    Robots whose samples are all dropped keep an empty list.
    """
    grouped: dict[int, list[tuple[float, int, Sample]]] = {}
    for row in track_rows:
        robot_id = int(row[0])
        bucket = grouped.setdefault(robot_id, [])
        position = field_position(float(row[3]), float(row[4]), field_size_ft, tolerance_ft)
        if position is None:
            continue
        conf = float(row[5]) if len(row) > 5 and row[5] is not None else None
        t_s = float(row[2])
        bucket.append((t_s, int(row[1]), (t_s, position[0], position[1], conf)))
    return {robot_id: [sample for _, _, sample in sorted(rows, key=lambda item: (item[0], item[1]))]
            for robot_id, rows in grouped.items()}


def debounced_zone_labels(samples: Sequence[Sample], zones: Mapping[str, Mapping[str, Any]],
                          debounce_s: float = DEFAULT_DEBOUNCE_S,
                          max_gap_s: float | None = DEFAULT_MAX_GAP_S) -> list[str | None]:
    """Per-sample debounced zone for one robot's time-sorted samples.

    Dwell debounce: a candidate zone (including "no zone") must be observed
    continuously for at least ``debounce_s`` of track time before the
    debounced zone switches. The switch is backdated to the sample where the
    candidate was first seen, so entry times do not depend on history. The
    debounced zone starts as "no zone", so the very first zone obeys the
    same dwell rule. A tracking gap longer than ``max_gap_s`` restarts the
    candidate's dwell (it was not observed during the gap).
    """
    labels: list[str | None] = []
    stable: str | None = None
    has_candidate = False
    candidate: str | None = None
    candidate_start = 0
    previous_t: float | None = None
    for index, (t_s, x_ft, y_ft, _conf) in enumerate(samples):
        raw = zone_at(x_ft, y_ft, zones)
        gap = previous_t is not None and max_gap_s is not None and t_s - previous_t > max_gap_s
        previous_t = t_s
        if raw == stable:
            has_candidate = False
        elif not has_candidate or raw != candidate or gap:
            has_candidate, candidate, candidate_start = True, raw, index
        if has_candidate and t_s - samples[candidate_start][0] >= debounce_s - 1e-9:
            stable, has_candidate = candidate, False
            for earlier in range(candidate_start, index):
                labels[earlier] = stable
        labels.append(stable)
    return labels
