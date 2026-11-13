from __future__ import annotations

import csv
import json
from pathlib import Path

from control_vision.pipeline.stats import robot_statistics


def export_stats_csv(path: Path, robots, tracks, cycles, zones: dict, confirmed_points: dict[int, int]):
    stats = robot_statistics(tracks, cycles, zones)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["team_number", "alliance", "cycles", "cycle_time_avg_s", "top_speed_ft_s", "distance_ft", "zone_time_s", "points_video"])
        writer.writeheader()
        for robot in robots:
            values = stats.get(robot["id"], {"cycles": 0, "cycle_time_avg_s": 0, "top_speed_ft_s": 0, "distance_ft": 0, "zone_time_s": {}})
            writer.writerow({"team_number": robot["team_number"], "alliance": robot["alliance"], "cycles": values["cycles"],
                             "cycle_time_avg_s": round(values["cycle_time_avg_s"], 3), "top_speed_ft_s": round(values["top_speed_ft_s"], 3),
                             "distance_ft": round(values["distance_ft"], 3), "zone_time_s": json.dumps(values["zone_time_s"], sort_keys=True),
                             "points_video": confirmed_points.get(robot["id"], 0)})


def export_scout_json(path: Path, video: dict, robots, tracks, cycles, events, clips, photos, zones: dict,
                      official_scores: dict[str, int | None] | None = None):
    stats = robot_statistics(tracks, cycles, zones)
    official_scores = official_scores or {"red": None, "blue": None}
    team_values = []
    for robot in robots:
        values = stats.get(robot["id"], {"cycles": 0, "cycle_time_avg_s": 0, "top_speed_ft_s": 0, "distance_ft": 0, "zone_time_s": {}})
        points = sum(event["points"] for event in events if event["robot_id"] == robot["id"] and event["status"] == "confirmed")
        team_values.append({"team_number": robot["team_number"], "alliance": robot["alliance"], "cycles": values["cycles"],
                            "cycle_time_avg_s": round(values["cycle_time_avg_s"], 3), "zone_time_s": values["zone_time_s"],
                            "top_speed_ft_s": round(values["top_speed_ft_s"], 3), "distance_ft": round(values["distance_ft"], 3),
                            "points_video": points, "points_official": None,
                            "confidence": values.get("confidence", "low")})
    payload = {"app": "control-vision", "version": "1.0.0", "event_code": video.get("event_code"),
               "match": video.get("match_number"), "video": Path(video["path"]).name,
               "teams": team_values, "official_alliance_scores": official_scores,
               "clips": [{"event_time_s": clip["t_s"], "robot": clip["team_number"], "points": clip["points"], "clip": Path(clip["path"]).name} for clip in clips],
               "photos": [{"team_number": photo["team_number"], "photo": Path(photo["path"]).name} for photo in photos]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
