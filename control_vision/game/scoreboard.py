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
                         disqualified: set[str] | None = None) -> dict:
    """Compute confirmed match points and BIOBUZZ ranking points.

    Events are evidence entries; this function never upgrades flagged events.
    Alliance attribution comes from scoring_alliance when set, otherwise from
    the robot's alliance. Disqualification is entered by the human reviewer.
    """
    events = list(events)
    robots = list(robots)
    disqualified = disqualified or set()
    match_resolved = bool(events) and all(event["status"] in {"confirmed", "rejected"} for event in events)
    robot_by_id = {int(robot["id"]): robot for robot in robots}
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


def calculate_playoff_scoreboard(events, robots, *, disqualified: set[str] | None = None,
                                 first_disqualified: str | None = None,
                                 simultaneous_disqualification: bool = False,
                                 match_final: bool = False) -> dict:
    """Score a playoff match without ranking points (BIOBUZZ T705).

    A human reviewer enters the chronological first DQ, or explicitly marks
    simultaneous DQs. Without that ruling, a double-DQ winner is unresolved.
    ``match_final`` allows a reviewed 0-0 match with no event rows to tie.
    """
    disqualified = set(disqualified or ())
    if disqualified - {"red", "blue"}:
        raise ValueError("Unknown playoff alliance DQ")
    if first_disqualified is not None and (len(disqualified) != 2 or first_disqualified not in disqualified):
        raise ValueError("First disqualified alliance requires both alliance DQs")
    if simultaneous_disqualification and (len(disqualified) != 2 or first_disqualified is not None):
        raise ValueError("Simultaneous DQ requires both alliances and no first DQ")

    events = list(events)
    reviewed = all(event["status"] in {"confirmed", "rejected"} for event in events)
    board = calculate_scoreboard(events, robots, disqualified=disqualified)
    for alliance in ("red", "blue"):
        for key in ("SWARM", "POLLINATOR_1", "POLLINATOR_2", "WIN_TIE", "total"):
            board["ranking_points"][alliance][key] = 0
    board["team_ranking_points"] = {robot_id: 0 for robot_id in board["team_ranking_points"]}

    winner = None
    tie = False
    resolved = False
    if len(disqualified) == 2:
        if simultaneous_disqualification:
            tie = resolved = True
        elif first_disqualified is not None:
            winner = "blue" if first_disqualified == "red" else "red"
            resolved = True
    elif len(disqualified) == 1:
        winner = "blue" if "red" in disqualified else "red"
        resolved = True
    elif match_final and reviewed:
        red, blue = board["alliance_points"]["red"], board["alliance_points"]["blue"]
        tie = red == blue
        winner = "red" if red > blue else "blue" if blue > red else None
        resolved = True
    # An official DQ can determine the winner while evidence reviews remain
    # pending. Keep review completion separate from the playoff outcome.
    board["match_resolved"] = board["match_resolved"] or (match_final and reviewed)
    board["playoff_outcome"] = {"resolved": resolved, "winner": winner, "tie": tie}
    return board
