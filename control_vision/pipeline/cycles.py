from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from control_vision.pipeline.zones import (DEFAULT_DEBOUNCE_S, DEFAULT_MAX_GAP_S, debounced_zone_labels,
                                           robot_samples)

PICKUP_ZONE_TYPES = frozenset({"pickup", "loading_zone"})
SCORING_ZONE_TYPES = frozenset({"scoring", "garden", "flower_scoring_volume", "hive"})


def count_zone_cycles(track_rows: Iterable[Sequence[Any]], zones: Mapping[str, Mapping[str, Any]],
                      debounce_s: float = DEFAULT_DEBOUNCE_S, *,
                      max_gap_s: float | None = DEFAULT_MAX_GAP_S) -> tuple[list[dict], list[dict]]:
    """Count pickup-to-scoring zone trips, returning cycles and provisional event candidates.

    Zones are the dwell-debounced zones from
    :func:`control_vision.pipeline.zones.debounced_zone_labels` (a zone
    counts only after ``debounce_s`` of continuous presence; entry is dated
    when the robot was first seen there). Per robot:
    EMPTY -> enters pickup -> AT_PICKUP -> enters scoring -> one cycle -> EMPTY.
    A cycle starts at the robot's first pickup entry after the previous
    cycle and ends at its scoring entry. Dwelling in or re-entering scoring
    without a new pickup visit never adds a cycle. Without ball detection
    these are zone trips, so cycles are medium confidence and their events
    are flagged, zero-point review evidence, never automatic points.
    """
    cycles: list[dict] = []
    events: list[dict] = []
    # Generic practice zones and BIOBUZZ user-drawn zones both produce
    # provisional trips.
    pickup_names = {name for name, zone in zones.items() if zone.get("type") in PICKUP_ZONE_TYPES}
    scoring_names = {name for name, zone in zones.items() if zone.get("type") in SCORING_ZONE_TYPES}
    for robot_id, samples in robot_samples(track_rows).items():
        labels = debounced_zone_labels(samples, zones, debounce_s, max_gap_s)
        state = "EMPTY"
        pickup_zone: str | None = None
        pickup_time: float | None = None
        previous_label: str | None = None
        for (t_s, _x, _y, _conf), label in zip(samples, labels):
            if label == previous_label:
                continue
            previous_label = label
            if state == "EMPTY" and label in pickup_names:
                state, pickup_zone, pickup_time = "AT_PICKUP", label, t_s
            elif state == "AT_PICKUP" and pickup_time is not None and label in scoring_names:
                cycles.append({"robot_id": robot_id, "start_t_s": pickup_time, "end_t_s": t_s,
                               "pickup_zone": pickup_zone, "scoring_zone": label, "confidence": "medium"})
                events.append({"robot_id": robot_id, "t_s": t_s, "points": 0, "confidence": "low",
                               "status": "flagged", "source": "zone_crossing"})
                state, pickup_zone, pickup_time = "EMPTY", None, None
    return cycles, events
