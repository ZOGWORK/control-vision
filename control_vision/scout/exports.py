from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from control_vision.game.scoreboard import calculate_playoff_scoreboard, calculate_scoreboard
from control_vision.pipeline.stats import empty_robot_statistics, robot_statistics

CSV_FIELDS = ["team_number", "alliance", "cycles", "cycle_time_avg_s", "top_speed_ft_s", "distance_ft",
              "zone_time_s", "points_video", "points_official", "ranking_points_video", "confidence",
              "match_type", "disqualified", "playoff_result"]
FINAL_STATUSES = frozenset({"confirmed", "rejected"})
MATCH_TYPES = frozenset({"qualification", "playoff"})


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


def match_type(video: Mapping[str, Any]) -> str:
    """Stored match type; a legacy row without one is a playoff only if it has an alliance DQ.

    The pre-match-type alliance-DQ checkboxes were labelled playoff-only, and
    the database migration applies the same rule to saved projects.
    """
    value = video.get("match_type")
    if value in (None, ""):
        return "playoff" if _disqualified(video) else "qualification"
    if value not in MATCH_TYPES:
        raise ValueError(f"Unknown match type: {value}")
    return str(value)


def match_ruling(video: Mapping[str, Any], robots: Iterable[Mapping[str, Any]],
                 qualification_disqualified_robot_ids: Iterable[int] | None = None) -> dict[str, Any]:
    """The referee ruling stored on the video and robot rows, checked for the match type."""
    kind = match_type(video)
    if qualification_disqualified_robot_ids is None:
        team_ids = {int(robot["id"]) for robot in robots if robot.get("qualification_disqualified")}
    else:
        team_ids = {int(robot_id) for robot_id in qualification_disqualified_robot_ids}
    alliances = _disqualified(video)
    first = video.get("playoff_first_disqualified") or None
    simultaneous = bool(video.get("playoff_simultaneous_dq"))
    final = bool(video.get("playoff_match_final"))
    if kind == "qualification" and (alliances or first or simultaneous):
        raise ValueError("A qualification match cannot carry a playoff alliance DQ or referee DQ ruling")
    if kind == "playoff" and team_ids:
        raise ValueError("A playoff match cannot carry a qualification team DQ")
    return {"match_type": kind, "qualification_disqualified_robot_ids": team_ids,
            "playoff_disqualified": alliances, "first_disqualified": first,
            "simultaneous_disqualification": simultaneous,
            "match_final": final if kind == "playoff" else False}


def match_scoreboard(video: Mapping[str, Any], robots: Iterable[Mapping[str, Any]], events: Iterable[Any],
                     qualification_disqualified_robot_ids: Iterable[int] | None = None) -> dict:
    """Score a match with the function its match type requires.

    Qualifications use ``calculate_scoreboard`` with team-level DQ (T601);
    playoffs use ``calculate_playoff_scoreboard`` with alliance DQ and the
    referee ruling (13.7, T705). No rule is re-implemented here. The board
    gains ``match_type``, ``ruling`` and, for playoffs, a
    ``pending_referee_ruling`` flag for a double DQ without a ruling.
    """
    robot_list = [dict(robot) for robot in robots]
    event_list = list(events)
    ruling = match_ruling(video, robot_list, qualification_disqualified_robot_ids)
    if ruling["match_type"] == "playoff":
        board = calculate_playoff_scoreboard(
            event_list, robot_list, disqualified=ruling["playoff_disqualified"],
            first_disqualified=ruling["first_disqualified"],
            simultaneous_disqualification=ruling["simultaneous_disqualification"],
            match_final=ruling["match_final"])
        board["pending_referee_ruling"] = (ruling["playoff_disqualified"] == {"red", "blue"}
                                           and not board["playoff_outcome"]["resolved"])
    else:
        team_dq = ({"qualification_disqualified_robot_ids": ruling["qualification_disqualified_robot_ids"]}
                   if ruling["qualification_disqualified_robot_ids"] else {})
        board = calculate_scoreboard(event_list, robot_list, _int_setting(video, "swarm_threshold", 16),
                                     _int_setting(video, "pollinator_1_threshold", 4),
                                     _int_setting(video, "pollinator_2_threshold", 7), **team_dq)
        board["pending_referee_ruling"] = False
    board["match_type"] = ruling["match_type"]
    board["ruling"] = ruling
    return board


def _scoreboard(video: Mapping[str, Any], robots: list[dict[str, Any]], events: list[dict[str, Any]],
                qualification_disqualified_robot_ids: set[int] | None = None) -> dict:
    """The single scoreboard every exported number is derived from."""
    return match_scoreboard(video, robots, events, qualification_disqualified_robot_ids)


def _ruling_payload(board: Mapping[str, Any], robots: list[dict[str, Any]]) -> dict[str, Any]:
    ruling = board["ruling"]
    teams = {int(robot["id"]): int(robot["team_number"]) for robot in robots}
    return {"qualification_disqualified_teams": sorted(teams[robot_id] for robot_id in
                                                       ruling["qualification_disqualified_robot_ids"]),
            "alliance_disqualified": sorted(ruling["playoff_disqualified"]),
            "first_disqualified": ruling["first_disqualified"],
            "simultaneous_disqualification": ruling["simultaneous_disqualification"],
            "match_final": ruling["match_final"]}


def _playoff_payload(board: Mapping[str, Any]) -> dict[str, Any] | None:
    if board["match_type"] != "playoff":
        return None
    outcome = board["playoff_outcome"]
    return {"resolved": outcome["resolved"], "winner": outcome["winner"], "tie": outcome["tie"],
            "pending_referee_ruling": board["pending_referee_ruling"]}


def _team_disqualified(board: Mapping[str, Any], robot: Mapping[str, Any]) -> bool:
    ruling = board["ruling"]
    return (int(robot["id"]) in ruling["qualification_disqualified_robot_ids"]
            or robot["alliance"] in ruling["playoff_disqualified"])


def _lookup(points: Mapping[Any, Any], robot_id: int) -> int:
    value = points.get(robot_id, points.get(str(robot_id), 0))
    return int(value or 0)


def _playoff_result(board: Mapping[str, Any], alliance: str) -> str | None:
    if board["match_type"] != "playoff":
        return None
    outcome = board["playoff_outcome"]
    if not outcome["resolved"]:
        return "pending"
    if outcome["tie"]:
        return "tie"
    return "win" if outcome["winner"] == alliance else "loss"


def _team_rows(robots: list[dict[str, Any]], tracks, cycles, zones: Mapping[str, Mapping[str, Any]],
               team_points: Mapping[Any, Any], ranking_points: Mapping[Any, Any],
               board: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Team rows shared by the CSV and Scout JSON so the two never fork.

    ``points_official`` is always None: official scores are per alliance and
    are never split into invented per-robot official points. With a match
    ``board``, rows also carry the match type, the team's DQ state and its
    playoff result; playoff rows have no ranking points (13.7).
    """
    stats = robot_statistics(tracks, cycles, zones)
    rows = []
    for robot in robots:
        robot_id = int(robot["id"])
        values = stats.get(robot_id) or empty_robot_statistics(zones)
        extra: dict[str, Any] = {}
        ranking: int | None = _lookup(ranking_points, robot_id)
        if board is not None:
            extra = {"match_type": board["match_type"], "disqualified": _team_disqualified(board, robot),
                     "playoff_result": _playoff_result(board, robot["alliance"])}
            if board["match_type"] == "playoff":
                ranking = None
        rows.append({"team_number": int(robot["team_number"]), "alliance": robot["alliance"],
                     "cycles": int(values["cycles"]),
                     "cycle_time_avg_s": round(float(values["cycle_time_avg_s"]), 3),
                     "zone_time_s": {name: round(float(seconds), 3) for name, seconds in values["zone_time_s"].items()},
                     "top_speed_ft_s": round(float(values["top_speed_ft_s"]), 3),
                     "distance_ft": round(float(values["distance_ft"]), 3),
                     "points_video": _lookup(team_points, robot_id), "points_official": None,
                     "ranking_points_video": ranking,
                     "confidence": values.get("confidence", "low"), **extra})
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
    board = None
    if events is not None:
        board = _scoreboard(_as_dict(video), robot_list, [_as_dict(event) for event in events],
                            qualification_disqualified_robot_ids)
        team_points, team_ranking = board["team_points"], board["team_ranking_points"]
    else:
        team_points, team_ranking = confirmed_points or {}, ranking_points or {}
    rows = _team_rows(robot_list, tracks, cycles, zones, team_points, team_ranking, board)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            ranking = row["ranking_points_video"]
            writer.writerow({**row, "zone_time_s": json.dumps(row["zone_time_s"], sort_keys=True),
                             "points_official": "", "ranking_points_video": "" if ranking is None else ranking,
                             "match_type": row.get("match_type") or "",
                             "disqualified": int(row["disqualified"]) if "disqualified" in row else "",
                             "playoff_result": row.get("playoff_result") or ""})


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
                                   board["team_ranking_points"], board),
               "official_alliance_scores": official,
               "scoreboard_ocr": scoreboard_ocr,
               # Playoffs award no Ranking Points (13.7), so RP fields are left out there.
               "alliance_results": {alliance: {"points_video": board["alliance_points"][alliance],
                                               **(board["ranking_points"][alliance]
                                                  if board["match_type"] == "qualification" else {})}
                                    for alliance in ("red", "blue")},
               "match_type": board["match_type"],
               "ruling": _ruling_payload(board, robot_list),
               "match_resolved": board["match_resolved"],
               "playoff_outcome": _playoff_payload(board),
               "clips": _clip_rows(clips, event_list, robot_list, video_row, qualification_disqualified_robot_ids),
               "photos": [{"team_number": photo["team_number"], "photo": _file_name(photo["path"])}
                          for photo in (_as_dict(photo) for photo in photos)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
