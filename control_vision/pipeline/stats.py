from __future__ import annotations

from math import hypot

from control_vision.pipeline.zones import zone_at


def robot_statistics(track_rows: list[tuple], cycles: list[dict], zones: dict) -> dict[int, dict]:
    grouped: dict[int, list[tuple]] = {}
    for row in track_rows:
        grouped.setdefault(int(row[0]), []).append(row)
    result = {}
    for robot_id, rows in grouped.items():
        rows.sort(key=lambda row: row[2])
        distance, speeds, zone_seconds = 0.0, [], {name: 0.0 for name in zones}
        previous_zone = None
        for first, second in zip(rows, rows[1:]):
            dt = max(0.0, float(second[2]) - float(first[2]))
            step = hypot(float(second[3]) - float(first[3]), float(second[4]) - float(first[4]))
            distance += step
            if dt > 0:
                speeds.append(step / dt)
            zone = zone_at(float(first[3]), float(first[4]), zones)
            if zone:
                zone_seconds[zone] += dt
        robot_cycles = [cycle for cycle in cycles if cycle["robot_id"] == robot_id]
        mean_cycle = (sum(c["end_t_s"] - c["start_t_s"] for c in robot_cycles) / len(robot_cycles)) if robot_cycles else 0.0
        mean_conf = sum(float(row[5]) for row in rows) / len(rows) if rows and len(rows[0]) > 5 else 0.0
        result[robot_id] = {"cycles": len(robot_cycles), "cycle_time_avg_s": mean_cycle,
                            "top_speed_ft_s": max(speeds, default=0.0), "distance_ft": distance,
                            "zone_time_s": zone_seconds, "confidence": "high" if mean_conf >= 0.7 else "medium" if mean_conf >= 0.4 else "low"}
    return result
