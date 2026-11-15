from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import hypot
from typing import Any

from control_vision.pipeline.zones import (DEFAULT_DEBOUNCE_S, DEFAULT_MAX_GAP_S, FIELD_SIZE_FT,
                                           FIELD_TOLERANCE_FT, Sample, debounced_zone_labels,
                                           field_zone_names, robot_samples)

SMOOTHING_WINDOW = 5
"""Moving-average window in frames on top-down positions (brief, pipeline step 7)."""

MAX_PLAUSIBLE_SPEED_FT_S = 15.0
"""Plausibility gate, not a measured robot limit: a step implying a faster
speed is treated as a tracking/ID jump and never integrated."""

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def empty_robot_statistics(zones: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Statistics for a robot with no usable track samples."""
    return {"cycles": 0, "cycle_time_avg_s": 0.0, "top_speed_ft_s": 0.0, "distance_ft": 0.0,
            "zone_time_s": {name: 0.0 for name in field_zone_names(zones)}, "confidence": "low"}


def _segments(samples: Sequence[Sample], max_gap_s: float, max_speed_ft_s: float) -> list[list[Sample]]:
    """Split a time-sorted track at tracking gaps and implausible jumps.

    Distance is only integrated inside a segment, so a gap longer than
    ``max_gap_s`` or a step faster than ``max_speed_ft_s`` (single-frame
    spike or ID swap) never contributes distance or speed. Duplicate
    timestamps keep the first sample.
    """
    segments: list[list[Sample]] = []
    for sample in samples:
        if segments:
            previous = segments[-1][-1]
            dt = sample[0] - previous[0]
            if dt <= 0:
                continue
            step = hypot(sample[1] - previous[1], sample[2] - previous[2])
            if dt <= max_gap_s and step / dt <= max_speed_ft_s:
                segments[-1].append(sample)
                continue
        segments.append([sample])
    return segments


def _moving_average(segment: Sequence[Sample], window: int) -> list[tuple[float, float, float]]:
    """Centred moving average; the window shrinks symmetrically at the ends."""
    half = max(0, (window - 1) // 2)
    smoothed = []
    for index, (t_s, _x, _y, _conf) in enumerate(segment):
        reach = min(half, index, len(segment) - 1 - index)
        part = segment[index - reach:index + reach + 1]
        smoothed.append((t_s, sum(s[1] for s in part) / len(part), sum(s[2] for s in part) / len(part)))
    return smoothed


def _tracking_confidence(samples: Sequence[Sample]) -> str:
    confs = [sample[3] for sample in samples if sample[3] is not None]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return "high" if mean_conf >= 0.7 else "medium" if mean_conf >= 0.4 else "low"


def robot_statistics(track_rows: Iterable[Sequence[Any]], cycles: Iterable[Mapping[str, Any]],
                     zones: Mapping[str, Mapping[str, Any]], *,
                     smoothing_window: int = SMOOTHING_WINDOW,
                     max_gap_s: float = DEFAULT_MAX_GAP_S,
                     max_speed_ft_s: float = MAX_PLAUSIBLE_SPEED_FT_S,
                     debounce_s: float = DEFAULT_DEBOUNCE_S,
                     field_size_ft: float = FIELD_SIZE_FT,
                     field_tolerance_ft: float = FIELD_TOLERANCE_FT) -> dict[int, dict[str, Any]]:
    """Per-robot cycles, average cycle time, top speed, distance and zone time.

    Samples are sorted per robot; positions outside the field by more than
    ``field_tolerance_ft`` are dropped, nearer ones clamped onto the field.
    Distance and speed use a ``smoothing_window``-frame moving average of
    positions inside segments split at gaps > ``max_gap_s`` and at steps
    faster than ``max_speed_ft_s``. Zone time uses the same dwell-debounced
    zones as :func:`control_vision.pipeline.cycles.count_zone_cycles` and
    never credits a tracking gap longer than ``max_gap_s``. Team confidence
    is the tracking confidence capped at medium unless every cycle is
    high confidence (zone trips without ball detection are medium).
    """
    cycle_list = [dict(cycle) for cycle in cycles]  # Mapping or sqlite3.Row
    zone_names = field_zone_names(zones)
    result: dict[int, dict[str, Any]] = {}
    for robot_id, samples in robot_samples(track_rows, field_size_ft, field_tolerance_ft).items():
        values = empty_robot_statistics(zones)
        distance, speeds = 0.0, []
        for segment in _segments(samples, max_gap_s, max_speed_ft_s):
            smoothed = _moving_average(segment, smoothing_window)
            for first, second in zip(smoothed, smoothed[1:]):
                dt = second[0] - first[0]
                step = hypot(second[1] - first[1], second[2] - first[2])
                if dt <= 0 or step / dt > max_speed_ft_s:
                    continue
                distance += step
                speeds.append(step / dt)
        zone_seconds = {name: 0.0 for name in zone_names}
        labels = debounced_zone_labels(samples, zones, debounce_s, max_gap_s)
        for index in range(len(samples) - 1):
            dt = samples[index + 1][0] - samples[index][0]
            zone = labels[index]
            if zone is not None and zone in zone_seconds and 0 < dt <= max_gap_s:
                zone_seconds[zone] += dt
        robot_cycles = [cycle for cycle in cycle_list if int(cycle["robot_id"]) == robot_id]
        mean_cycle = (sum(float(c["end_t_s"]) - float(c["start_t_s"]) for c in robot_cycles) / len(robot_cycles)
                      if robot_cycles else 0.0)
        confidence = _tracking_confidence(samples)
        ball_confirmed = bool(robot_cycles) and all(cycle.get("confidence") == "high" for cycle in robot_cycles)
        if not ball_confirmed and _CONFIDENCE_RANK[confidence] > _CONFIDENCE_RANK["medium"]:
            confidence = "medium"
        values.update({"cycles": len(robot_cycles), "cycle_time_avg_s": float(mean_cycle),
                       "top_speed_ft_s": float(max(speeds, default=0.0)), "distance_ft": float(distance),
                       "zone_time_s": zone_seconds, "confidence": confidence if samples else "low"})
        result[robot_id] = values
    return result
