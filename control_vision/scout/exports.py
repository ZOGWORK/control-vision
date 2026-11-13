from __future__ import annotations

import csv
import json
from pathlib import Path

from control_vision.pipeline.stats import robot_statistics


def export_stats_csv(path: Path, robots, tracks, cycles, zones: dict, confirmed_points: dict[int, int], ranking_points: dict[int, int] | None = None):
    stats = robot_statistics(tracks, cycles, zones)
    ranking_points = ranking_points or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["team_number", "alliance", "cycles", "cycle_time_avg_s", "top_speed_ft_s", "distance_ft", "zone_time_s", "points_video", "ranking_points_video"])
        writer.writeheader()
        for robot in robots:
            values = stats.get(robot["id"], {"cycles": 0, "cycle_time_avg_s": 0, "top_speed_ft_s": 0, "distance_ft": 0, "zone_time_s": {}})
            writer.writerow({"team_number": robot["team_number"], "alliance": robot["alliance"], "cycles": values["cycles"],
                             "cycle_time_avg_s": round(values["cycle_time_avg_s"], 3), "top_speed_ft_s": round(values["top_speed_ft_s"], 3),
                             "distance_ft": round(values["distance_ft"], 3), "zone_time_s": json.dumps(values["zone_time_s"], sort_keys=True),
                             "points_video": confirmed_points.get(robot["id"], 0),
                             "ranking_points_video": ranking_points.get(int(robot["id"]), 0)})


def export_scout_json(path: Path, video: dict, robots, tracks, cycles, events, clips, photos, zones: dict,
                      official_scores: dict[str, int | None] | None = None):
    stats = robot_statistics(tracks, cycles, zones)
    from control_vision.game.scoreboard import calculate_scoreboard
    disqualified = {alliance for alliance, key in (("red", "red_disqualified"), ("blue", "blue_disqualified")) if video.get(key)}
    board = calculate_scoreboard(events, robots, int(video.get("swarm_threshold", 16)),
                                 int(video.get("pollinator_1_threshold", 4)), int(video.get("pollinator_2_threshold", 7)), disqualified)
    if official_scores is None:
        official_scores = {"red": video.get("official_red"), "blue": video.get("official_blue")}
    try:
        scoreboard_ocr = json.loads(video.get("scoreboard_ocr_json") or "null")
    except (TypeError, json.JSONDecodeError):
        scoreboard_ocr = None
    team_values = []
    for robot in robots:
        values = stats.get(robot["id"], {"cycles": 0, "cycle_time_avg_s": 0, "top_speed_ft_s": 0, "distance_ft": 0, "zone_time_s": {}})
        points = board["team_points"].get(int(robot["id"]), 0)
        team_values.append({"team_number": robot["team_number"], "alliance": robot["alliance"], "cycles": values["cycles"],
                            "cycle_time_avg_s": round(values["cycle_time_avg_s"], 3), "zone_time_s": values["zone_time_s"],
                            "top_speed_ft_s": round(values["top_speed_ft_s"], 3), "distance_ft": round(values["distance_ft"], 3),
                            "points_video": points, "points_official": None,
                            "ranking_points_video": board["team_ranking_points"].get(int(robot["id"]), 0),
                            "confidence": values.get("confidence", "low")})
    payload = {"app": "control-vision", "version": "1.0.0", "event_code": video.get("event_code"),
               "match": video.get("match_number"), "video": Path(video["path"]).name,
               "teams": team_values, "official_alliance_scores": official_scores,
               "scoreboard_ocr": scoreboard_ocr,
               "alliance_results": {alliance: {"points_video": board["alliance_points"][alliance], **board["ranking_points"][alliance]} for alliance in ("red", "blue")},
               "clips": [{"event_time_s": clip["t_s"], "robot": clip["team_number"], "points": clip["points"], "clip": Path(clip["path"]).name} for clip in clips],
               "photos": [{"team_number": photo["team_number"], "photo": Path(photo["path"]).name} for photo in photos]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
