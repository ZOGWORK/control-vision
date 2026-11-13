from __future__ import annotations

def _optional_field(row, name: str):
    """Read an optional field from dicts and sqlite3.Row objects."""
    try:
        return row[name]
    except (KeyError, IndexError):
        return None


def calculate_scoreboard(events, robots, swarm_threshold: int = 16,
                         pollinator_1_threshold: int = 4,
                         pollinator_2_threshold: int = 7,
                         disqualified: set[str] | None = None,
                         qualification_disqualified_robot_ids: set[int] | None = None) -> dict:
    """Compute confirmed match points and BIOBUZZ ranking points.

    Events are evidence entries; this function never upgrades flagged events.
    Alliance attribution comes from scoring_alliance when set, otherwise from
    the robot's alliance. Disqualification is entered by the human reviewer.
    ``disqualified`` is playoff alliance DQ. Qualification DQ uses robot IDs
    and zeroes only the affected team's match points and ranking points.
    ``team_points`` attributes observed actions to robots; ``team_match_points``
    is the official alliance match score credited to each eligible team.
    """
    events = list(events)
    robots = list(robots)
    disqualified = disqualified or set()
    qualification_disqualified = {int(robot_id) for robot_id in
                                   (qualification_disqualified_robot_ids or set())}
    if disqualified and qualification_disqualified:
        raise ValueError("Playoff alliance DQ and qualification team DQ cannot be combined")
    match_resolved = bool(events) and all(event["status"] in {"confirmed", "rejected"} for event in events)
    robot_by_id = {int(robot["id"]): robot for robot in robots}
    unknown_ids = qualification_disqualified - robot_by_id.keys()
    if unknown_ids:
        raise ValueError(f"Unknown qualification DQ robot IDs: {sorted(unknown_ids)}")
    alliance_scores = {"red": 0, "blue": 0}
    team_points: dict[int, int] = {int(robot["id"]): 0 for robot in robots}
    swarm_points = {"red": 0, "blue": 0}
    hive_tips = {"red": 0, "blue": 0}

    for event in events:
        if event["status"] != "confirmed":
            continue
        robot_id = event["robot_id"]
        robot = robot_by_id.get(int(robot_id)) if robot_id is not None else None
        alliance = (_optional_field(event, "scoring_alliance") or _optional_field(event, "event_alliance")
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
        # Robot attribution describes the observed actor; the recipient
        # alliance can differ for alliance-scoped achievements and fouls.
        if robot:
            team_points[int(robot_id)] += points

    for alliance in disqualified:
        if alliance in alliance_scores:
            alliance_scores[alliance] = 0
            for robot in robots:
                if robot["alliance"] == alliance:
                    team_points[int(robot["id"])] = 0

    for robot_id in qualification_disqualified:
        team_points[robot_id] = 0

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
        elif ("blue" if alliance == "red" else "red") in disqualified:
            outcome = 3
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
    team_match_points = {}
    for robot in robots:
        robot_id = int(robot["id"])
        team_rp[robot_id] = (0 if robot_id in qualification_disqualified else
                             ranking_points[robot["alliance"]]["total"])
        team_match_points[robot_id] = (0 if robot_id in qualification_disqualified else
                                       alliance_scores[robot["alliance"]])
    return {"alliance_points": alliance_scores, "team_points": dict(team_points),
            "ranking_points": ranking_points, "team_ranking_points": team_rp,
            "team_match_points": team_match_points,
            "swarm_threshold": swarm_threshold,
            "pollinator_thresholds": [pollinator_1_threshold, pollinator_2_threshold],
            "disqualified": sorted(disqualified),
            "qualification_disqualified_robot_ids": sorted(qualification_disqualified),
            "match_resolved": match_resolved}


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
