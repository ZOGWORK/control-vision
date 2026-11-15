from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from control_vision.game.scoreboard import calculate_scoreboard
from control_vision.pipeline.stats import empty_robot_statistics, robot_statistics

CSV_FIELDS = ["team_number", "alliance", "cycles", "cycle_time_avg_s", "top_speed_ft_s", "distance_ft",
              "zone_time_s", "points_video", "points_official", "ranking_points_video", "confidence"]
FINAL_STATUSES = frozenset({"confirmed", "rejected"})


def _as_dict(row: Any) -> dict[str, Any]:
    """Accept a Mapping or a sqlite3.Row (anything with keys())."""
    return {} if row is None else dict(row)


def _file_name(path: Any) -> str | None:
    """File name only (never the full path); handles / and \\ separators."""
    if path is None or str(path) == "":
        return None
    return PureWindowsPath(str(path)).name


def _int_setting(video: Mapping[str, Any], key: str, default: int) -> int:
    value = video.get(key)
    return default if value is None or value == "" else int(value)


def _disqualified(video: Mapping[str, Any]) -> set[str]:
    return {alliance for alliance, key in (("red", "red_disqualified"), ("blue", "blue_disqualified"))
            if video.get(key)}


def _scoreboard(video: Mapping[str, Any], robots: list[dict[str, Any]], events: list[dict[str, Any]],
                qualification_disqualified_robot_ids: set[int] | None = None) -> dict:
    """The single scoreboard every exported number is derived from."""
    # Pass team DQ only when present so callers on an older scoreboard keep working.
    team_dq = ({"qualification_disqualified_robot_ids": qualification_disqualified_robot_ids}
               if qualification_disqualified_robot_ids else {})
    return calculate_scoreboard(events, robots, _int_setting(video, "swarm_threshold", 16),
                                _int_setting(video, "pollinator_1_threshold", 4),
                                _int_setting(video, "pollinator_2_threshold", 7), _disqualified(video), **team_dq)


def _lookup(points: Mapping[Any, Any], robot_id: int) -> int:
    value = points.get(robot_id, points.get(str(robot_id), 0))
    return int(value or 0)


def _team_rows(robots: list[dict[str, Any]], tracks, cycles, zones: Mapping[str, Mapping[str, Any]],
               team_points: Mapping[Any, Any], ranking_points: Mapping[Any, Any]) -> list[dict[str, Any]]:
    """Team rows shared by the CSV and Scout JSON so the two never fork.

    ``points_official`` is always None: official scores are per alliance and
    are never split into invented per-robot official points.
    """
    stats = robot_statistics(tracks, cycles, zones)
    rows = []
    for robot in robots:
        robot_id = int(robot["id"])
        values = stats.get(robot_id) or empty_robot_statistics(zones)
        rows.append({"team_number": int(robot["team_number"]), "alliance": robot["alliance"],
                     "cycles": int(values["cycles"]),
                     "cycle_time_avg_s": round(float(values["cycle_time_avg_s"]), 3),
                     "zone_time_s": {name: round(float(seconds), 3) for name, seconds in values["zone_time_s"].items()},
                     "top_speed_ft_s": round(float(values["top_speed_ft_s"]), 3),
                     "distance_ft": round(float(values["distance_ft"]), 3),
                     "points_video": _lookup(team_points, robot_id), "points_official": None,
                     "ranking_points_video": _lookup(ranking_points, robot_id),
                     "confidence": values.get("confidence", "low")})
    return rows


def _clip_rows(clips: Iterable[Any], events: list[dict[str, Any]], robots: list[dict[str, Any]],
               video: Mapping[str, Any], qualification_disqualified_robot_ids: set[int] | None) -> list[dict[str, Any]]:
    """Clip rows whose points are the credit calculate_scoreboard gives the event.

    ``recipient_alliance`` is the alliance the event's points go to (the
    opponent for a foul). ``points`` is what that alliance is credited and
    ``robot_points`` what the observed robot is credited (0 for fouls and
    manual adjustments); both respect alliance and team DQ. Points are None
    for events still awaiting review and 0 for rejected events.
    """
    events_by_id = {int(event["id"]): event for event in events if event.get("id") is not None}
    teams = {int(robot["id"]): robot["team_number"] for robot in robots}
    rows = []
    for clip in (_as_dict(clip) for clip in clips):
        event_id = clip.get("scoring_event_id")
        event = dict(events_by_id.get(int(event_id), clip) if event_id is not None else clip)
        status = event.get("status") or "unknown"
        robot_id = event.get("robot_id")
        robot_key = int(robot_id) if robot_id is not None else None
        # Probe with a one-point confirmed copy and no DQ to learn where the
        # scoreboard sends this event's points, without re-implementing rules.
        probe = calculate_scoreboard([dict(event, status="confirmed", points=1)], robots)
        recipient = next((alliance for alliance in ("red", "blue") if probe["alliance_points"][alliance]), None)
        points: int | None
        robot_points: int | None
        if status == "confirmed":
            credit = _scoreboard(video, robots, [event], qualification_disqualified_robot_ids)
            points = int(credit["alliance_points"][recipient]) if recipient else 0
            robot_points = int(credit["team_points"].get(robot_key, 0)) if robot_key is not None else 0
        elif status == "rejected":
            points = robot_points = 0
        else:
            points = robot_points = None
        team_number = teams.get(robot_key) if robot_key is not None else None
        if team_number is None:
            team_number = clip.get("team_number")
        rows.append({"event_time_s": float(event.get("t_s", clip.get("t_s")) or 0.0), "robot": team_number,
                     "team_number": team_number, "status": status,
                     "event_type": event.get("event_type") or "unclassified",
                     "confidence": event.get("confidence"), "recipient_alliance": recipient,
                     "points": points, "robot_points": robot_points, "clip": _file_name(clip.get("path"))})
    return rows


def export_stats_csv(path: Path, robots, tracks, cycles, zones: dict, confirmed_points: Mapping[Any, Any] | None = None,
                     ranking_points: Mapping[Any, Any] | None = None, *, events: Iterable[Any] | None = None,
                     video: Any = None, qualification_disqualified_robot_ids: set[int] | None = None) -> None:
    """Write per-team statistics as CSV.

    Pass ``events`` (and the ``video`` row for DQ and thresholds) so points
    come from the same calculate_scoreboard result as the Scout JSON. The
    legacy ``confirmed_points``/``ranking_points`` maps are used only when
    ``events`` is omitted; they must come from calculate_scoreboard too.
    """
    robot_list = [_as_dict(robot) for robot in robots]
    if events is not None:
        board = _scoreboard(_as_dict(video), robot_list, [_as_dict(event) for event in events],
                            qualification_disqualified_robot_ids)
        team_points, team_ranking = board["team_points"], board["team_ranking_points"]
    else:
        team_points, team_ranking = confirmed_points or {}, ranking_points or {}
    rows = _team_rows(robot_list, tracks, cycles, zones, team_points, team_ranking)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "zone_time_s": json.dumps(row["zone_time_s"], sort_keys=True),
                             "points_official": ""})


def export_scout_json(path: Path, video: Any, robots, tracks, cycles, events, clips, photos, zones: dict,
                      official_scores: Mapping[str, int | None] | None = None, *,
                      qualification_disqualified_robot_ids: set[int] | None = None) -> None:
    """Write the Control Scout v1 match JSON.

    ``video`` may be a Mapping or a sqlite3.Row. Official alliance scores
    missing from ``official_scores`` fall back to the stored video row.
    """
    video_row = _as_dict(video)
    robot_list = [_as_dict(robot) for robot in robots]
    event_list = [_as_dict(event) for event in events]
    board = _scoreboard(video_row, robot_list, event_list, qualification_disqualified_robot_ids)
    provided = dict(official_scores or {})
    stored = {"red": video_row.get("official_red"), "blue": video_row.get("official_blue")}
    official = {alliance: provided[alliance] if alliance in provided else stored[alliance]
                for alliance in ("red", "blue")}
    try:
        scoreboard_ocr = json.loads(video_row.get("scoreboard_ocr_json") or "null")
    except (TypeError, json.JSONDecodeError):
        scoreboard_ocr = None
    payload = {"app": "control-vision", "version": "1.0.0", "event_code": video_row.get("event_code"),
               "match": video_row.get("match_number"), "video": _file_name(video_row.get("path")),
               "teams": _team_rows(robot_list, tracks, cycles, zones, board["team_points"],
                                   board["team_ranking_points"]),
               "official_alliance_scores": official,
               "scoreboard_ocr": scoreboard_ocr,
               "alliance_results": {alliance: {"points_video": board["alliance_points"][alliance],
                                               **board["ranking_points"][alliance]} for alliance in ("red", "blue")},
               "clips": _clip_rows(clips, event_list, robot_list, video_row, qualification_disqualified_robot_ids),
               "photos": [{"team_number": photo["team_number"], "photo": _file_name(photo["path"])}
                          for photo in (_as_dict(photo) for photo in photos)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
