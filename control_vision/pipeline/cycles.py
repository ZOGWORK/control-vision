from __future__ import annotations

from control_vision.pipeline.zones import zone_at


def count_zone_cycles(track_rows: list[tuple], zones: dict, debounce_s: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Count pickup-to-scoring zone trips, returning cycles and provisional event candidates."""
    by_robot: dict[int, list[tuple]] = {}
    for row in track_rows:
        by_robot.setdefault(int(row[0]), []).append(row)
    cycles, events = [], []
    pickup_names = {name for name, zone in zones.items() if zone.get("type") == "pickup"}
    scoring_names = {name for name, zone in zones.items() if zone.get("type") == "scoring"}
    for robot_id, rows in by_robot.items():
        rows.sort(key=lambda row: float(row[2]))
        state = "EMPTY"
        pickup_zone = None
        pickup_time = None
        last_zone, last_change = None, -float("inf")
        for row in rows:
            _, _, t_s, x_ft, y_ft, conf = row
            current_zone = zone_at(float(x_ft), float(y_ft), zones)
            if current_zone != last_zone:
                if float(t_s) - last_change < debounce_s:
                    current_zone = last_zone
                else:
                    last_zone, last_change = current_zone, float(t_s)
            if state == "EMPTY" and current_zone in pickup_names:
                state, pickup_zone, pickup_time = "AT_PICKUP", current_zone, float(t_s)
            elif state == "AT_PICKUP" and current_zone in scoring_names and float(t_s) > pickup_time:
                end_time = float(t_s)
                cycles.append({"robot_id": robot_id, "start_t_s": pickup_time, "end_t_s": end_time,
                               "pickup_zone": pickup_zone, "scoring_zone": current_zone, "confidence": "medium"})
                events.append({"robot_id": robot_id, "t_s": end_time, "points": 0, "confidence": "low",
                               "status": "flagged", "source": "zone_crossing"})
                state, pickup_zone, pickup_time = "EMPTY", None, None
    return cycles, events
