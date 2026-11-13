from __future__ import annotations

from collections import defaultdict


def _optional_field(row, name: str):
    """Read an optional field from dicts and sqlite3.Row objects."""
    try:
        return row[name]
    except (KeyError, IndexError):
        return None


def calculate_scoreboard(events, robots, swarm_threshold: int = 16,
                         pollinator_1_threshold: int = 4,
                         pollinator_2_threshold: int = 7,
                         disqualified: set[str] | None = None) -> dict:
    """Compute confirmed match points and BIOBUZZ ranking points.

    Events are evidence entries; this function never upgrades flagged events.
    Alliance attribution comes from scoring_alliance when set, otherwise from
    the robot's alliance. Disqualification is entered by the human reviewer.
    """
    disqualified = disqualified or set()
    match_resolved = all(event["status"] in {"confirmed", "rejected"} for event in events)
    robot_by_id = {int(robot["id"]): robot for robot in robots}
    alliance_scores = {"red": 0, "blue": 0}
    team_points: dict[int, int] = defaultdict(int)
    swarm_points = {"red": 0, "blue": 0}
    hive_tips = {"red": 0, "blue": 0}

    for event in events:
        if event["status"] != "confirmed":
            continue
        robot_id = event["robot_id"]
        robot = robot_by_id.get(int(robot_id)) if robot_id is not None else None
        alliance = (_optional_field(event, "event_alliance") or _optional_field(event, "scoring_alliance")
                    or (robot["alliance"] if robot else None))
        if alliance not in alliance_scores:
            continue
        points = int(event["points"])
        alliance_scores[alliance] += points
        event_type = _optional_field(event, "event_type") or "unclassified"
        if event_type in {"leave_auto", "park_auto", "park_teleop"}:
            swarm_points[alliance] += points
        if event_type in {"hive_tip_auto", "hive_tip_teleop"}:
            hive_tips[alliance] += 1
        if robot and robot["alliance"] == alliance:
            team_points[int(robot_id)] += points

    for alliance in disqualified:
        if alliance in alliance_scores:
            alliance_scores[alliance] = 0
            for robot in robots:
                if robot["alliance"] == alliance:
                    team_points[int(robot["id"])] = 0

    ranking_points = {"red": {}, "blue": {}}
    for alliance in ("red", "blue"):
        swarm = int(swarm_points[alliance] >= swarm_threshold)
        p1 = int(hive_tips[alliance] >= pollinator_1_threshold)
        p2 = int(hive_tips[alliance] >= pollinator_2_threshold)
        if alliance in disqualified:
            swarm = p1 = p2 = 0
            outcome = 0
        elif not match_resolved:
            outcome = 0
        elif alliance_scores[alliance] > alliance_scores["blue" if alliance == "red" else "red"]:
            outcome = 3
        elif alliance_scores[alliance] == alliance_scores["blue" if alliance == "red" else "red"]:
            outcome = 1
        else:
            outcome = 0
        ranking_points[alliance] = {
            "SWARM": swarm,
            "POLLINATOR_1": p1,
            "POLLINATOR_2": p2,
            "WIN_TIE": outcome,
            "total": swarm + p1 + p2 + outcome,
            "swarm_threshold_progress": swarm_points[alliance],
            "hive_tips": hive_tips[alliance],
        }

    team_rp = {}
    for robot in robots:
        team_rp[int(robot["id"])] = ranking_points[robot["alliance"]]["total"]
    return {"alliance_points": alliance_scores, "team_points": dict(team_points),
            "ranking_points": ranking_points, "team_ranking_points": team_rp,
            "swarm_threshold": swarm_threshold,
            "pollinator_thresholds": [pollinator_1_threshold, pollinator_2_threshold],
            "disqualified": sorted(disqualified), "match_resolved": match_resolved}


def classify_match_time(t_s: float, periods: dict) -> tuple[str, float]:
    """Return match phase and countdown-style clock for elapsed source time."""
    auto = float(periods["auto"])
    transition = float(periods["transition"])
    teleop = float(periods["teleop"])
    if t_s < auto:
        return "AUTO", auto - max(0.0, t_s)
    if t_s < auto + transition:
        return "TRANSITION", auto + transition - t_s
    if t_s < auto + transition + teleop:
        return "TELEOP", auto + transition + teleop - t_s
    return "POST-MATCH", 0.0
