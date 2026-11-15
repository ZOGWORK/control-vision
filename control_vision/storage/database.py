from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS videos (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id), path TEXT NOT NULL, duration_s REAL, fps REAL, width INTEGER, height INTEGER, event_code TEXT, match_number TEXT, official_red INTEGER, official_blue INTEGER, red_disqualified INTEGER NOT NULL DEFAULT 0, blue_disqualified INTEGER NOT NULL DEFAULT 0, swarm_threshold INTEGER NOT NULL DEFAULT 16, pollinator_1_threshold INTEGER NOT NULL DEFAULT 4, pollinator_2_threshold INTEGER NOT NULL DEFAULT 7, scoreboard_ocr_json TEXT, tracking_mode TEXT NOT NULL DEFAULT 'yolo', analysis_status TEXT NOT NULL DEFAULT 'unprocessed');
CREATE TABLE IF NOT EXISTS calibrations (video_id INTEGER PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE, corners_json TEXT NOT NULL, zones_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS robots (id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE, team_number INTEGER NOT NULL, alliance TEXT NOT NULL CHECK(alliance IN ('red','blue')), color_hint TEXT, seed_x INTEGER, seed_y INTEGER);
CREATE TABLE IF NOT EXISTS tracks (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, frame INTEGER NOT NULL, t_s REAL NOT NULL, x_ft REAL NOT NULL, y_ft REAL NOT NULL, conf REAL NOT NULL, UNIQUE(robot_id, frame));
CREATE TABLE IF NOT EXISTS cycles (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, start_t_s REAL NOT NULL, end_t_s REAL NOT NULL, pickup_zone TEXT, scoring_zone TEXT);
CREATE TABLE IF NOT EXISTS scoring_events (id INTEGER PRIMARY KEY, robot_id INTEGER REFERENCES robots(id) ON DELETE SET NULL, video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE, origin_robot_id INTEGER, t_s REAL NOT NULL, points INTEGER NOT NULL DEFAULT 0, confidence TEXT NOT NULL DEFAULT 'low', status TEXT NOT NULL DEFAULT 'flagged', source TEXT NOT NULL DEFAULT 'zone_crossing', event_type TEXT NOT NULL DEFAULT 'unclassified', scoring_alliance TEXT CHECK(scoring_alliance IN ('red','blue')));
CREATE TABLE IF NOT EXISTS clips (id INTEGER PRIMARY KEY, scoring_event_id INTEGER REFERENCES scoring_events(id) ON DELETE CASCADE, path TEXT NOT NULL, start_t_s REAL NOT NULL, end_t_s REAL NOT NULL);
CREATE TABLE IF NOT EXISTS photos (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, path TEXT NOT NULL, t_s REAL NOT NULL);
"""

_UNCHANGED: Any = object()
GENERATED_SOURCES = frozenset({"zone_crossing", "goal_roi"})
EVENT_STATUSES = frozenset({"confirmed", "flagged", "rejected"})
CONFIDENCES = frozenset({"high", "medium", "low"})
ALLIANCES = frozenset({"red", "blue"})


def _inserted_id(cursor: sqlite3.Cursor) -> int:
    row_id = cursor.lastrowid
    if row_id is None:
        raise RuntimeError("Database insert did not return a row ID")
    return int(row_id)


def _match_preserved_events(events: list[dict], preserved: Sequence[Any]) -> dict[int, int]:
    """Match same-source candidates by maximum count, then minimum time error."""
    matched: dict[int, int] = {}
    keys = {(event.get("robot_id") if event.get("robot_id") is not None else 0,
             event.get("source", "zone_crossing")) for event in events}
    for origin, source in keys:
        candidates = sorted(((index, float(event["t_s"])) for index, event in enumerate(events)
                             if (event.get("robot_id") if event.get("robot_id") is not None else 0,
                                 event.get("source", "zone_crossing")) == (origin, source)),
                            key=lambda item: item[1])
        rows = sorted(((int(row["id"]), float(row["t_s"])) for row in preserved
                       if (row["origin_robot_id"], row["source"]) == (origin, source)),
                      key=lambda item: item[1])
        if not candidates or not rows:
            continue
        count, width = len(candidates), len(rows)
        scores: list[list[tuple[int, float]]] = [[(0, 0.0) for _ in range(width + 1)]
                                                  for _ in range(count + 1)]
        choices = [["" for _ in range(width + 1)] for _ in range(count + 1)]
        for i in range(1, count + 1):
            for j in range(1, width + 1):
                best = scores[i - 1][j]
                choice = "candidate"
                skip_row = scores[i][j - 1]
                if (skip_row[0], -skip_row[1]) > (best[0], -best[1]):
                    best, choice = skip_row, "row"
                distance = abs(candidates[i - 1][1] - rows[j - 1][1])
                if distance <= 0.5:
                    previous = scores[i - 1][j - 1]
                    paired = (previous[0] + 1, previous[1] + distance)
                    if (paired[0], -paired[1]) >= (best[0], -best[1]):
                        best, choice = paired, "pair"
                scores[i][j], choices[i][j] = best, choice
        i, j = count, width
        while i and j:
            choice = choices[i][j]
            if choice == "pair":
                matched[candidates[i - 1][0]] = rows[j - 1][0]
                i -= 1
                j -= 1
            elif choice == "candidate":
                i -= 1
            else:
                j -= 1
    return matched


class ProjectDatabase:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.executescript(SCHEMA)
            # One explicit transaction: SQLite DDL is transactional, so a
            # failure part-way leaves the old schema intact for a clean retry.
            self.connection.execute("BEGIN IMMEDIATE")
            self._migrate()
            self.connection.commit()
        except BaseException:
            if self.connection.in_transaction:
                self.connection.rollback()
            self.connection.close()
            raise

    def _migrate(self) -> None:
        video_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(videos)")}
        added_analysis_status = "analysis_status" not in video_columns
        for column in ("official_red", "official_blue", "red_disqualified", "blue_disqualified",
                       "swarm_threshold", "pollinator_1_threshold", "pollinator_2_threshold", "scoreboard_ocr_json", "tracking_mode", "analysis_status"):
            if column not in video_columns:
                if column == "scoreboard_ocr_json":
                    self.connection.execute("ALTER TABLE videos ADD COLUMN scoreboard_ocr_json TEXT")
                elif column == "tracking_mode":
                    self.connection.execute("ALTER TABLE videos ADD COLUMN tracking_mode TEXT NOT NULL DEFAULT 'yolo'")
                elif column == "analysis_status":
                    self.connection.execute("ALTER TABLE videos ADD COLUMN analysis_status TEXT NOT NULL DEFAULT 'unprocessed'")
                else:
                    default = " NOT NULL DEFAULT 0" if column.endswith("disqualified") else " NOT NULL DEFAULT 16" if column == "swarm_threshold" else " NOT NULL DEFAULT 4" if column == "pollinator_1_threshold" else " NOT NULL DEFAULT 7" if column == "pollinator_2_threshold" else ""
                    self.connection.execute(f"ALTER TABLE videos ADD COLUMN {column} INTEGER{default}")
        robot_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(robots)")}
        for column in ("seed_x", "seed_y"):
            if column not in robot_columns:
                self.connection.execute(f"ALTER TABLE robots ADD COLUMN {column} INTEGER")
        cycle_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(cycles)")}
        if "confidence" not in cycle_columns:
            self.connection.execute("ALTER TABLE cycles ADD COLUMN confidence TEXT NOT NULL DEFAULT 'medium'")
        event_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(scoring_events)")}
        if "event_type" not in event_columns:
            self.connection.execute("ALTER TABLE scoring_events ADD COLUMN event_type TEXT NOT NULL DEFAULT 'unclassified'")
        if "scoring_alliance" not in event_columns:
            self.connection.execute("ALTER TABLE scoring_events ADD COLUMN scoring_alliance TEXT")
        if "video_id" not in event_columns:
            self.connection.execute("ALTER TABLE scoring_events ADD COLUMN video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE")
        # Idempotent backfill for robot-owned events that predate video_id.
        # Spec-era tables had no foreign keys, so skip robots whose video row
        # is gone instead of failing the whole project open.
        self.connection.execute(
            "UPDATE scoring_events SET video_id=(SELECT r.video_id FROM robots r WHERE r.id=scoring_events.robot_id) "
            "WHERE video_id IS NULL AND robot_id IS NOT NULL AND EXISTS "
            "(SELECT 1 FROM robots r JOIN videos v ON v.id=r.video_id WHERE r.id=scoring_events.robot_id)")
        if "origin_robot_id" not in event_columns:
            self.connection.execute("ALTER TABLE scoring_events ADD COLUMN origin_robot_id INTEGER")
        if added_analysis_status:
            # Old projects have no processing marker. Preserve access to videos
            # with saved evidence while leaving empty imports unprocessed.
            self.connection.execute(
                "UPDATE videos SET analysis_status='ready' WHERE "
                "EXISTS (SELECT 1 FROM tracks t JOIN robots r ON r.id=t.robot_id WHERE r.video_id=videos.id) "
                "OR EXISTS (SELECT 1 FROM cycles c JOIN robots r ON r.id=c.robot_id WHERE r.video_id=videos.id) "
                "OR EXISTS (SELECT 1 FROM scoring_events e WHERE e.video_id=videos.id)"
            )

    def add_project(self, name: str) -> int:
        cursor = self.connection.execute("INSERT INTO projects(name) VALUES (?)", (name,))
        self.connection.commit()
        return _inserted_id(cursor)

    def add_video(self, project_id: int, path: Path, metadata: dict[str, Any], event_code: str | None = None, match_number: str | None = None) -> int:
        cursor = self.connection.execute(
            "INSERT INTO videos(project_id,path,duration_s,fps,width,height,event_code,match_number,analysis_status) VALUES (?,?,?,?,?,?,?,?,?)",
            (project_id, str(path), metadata["duration_s"], metadata["fps"], metadata["width"], metadata["height"], event_code, match_number, "unprocessed"),
        )
        self.connection.commit()
        return _inserted_id(cursor)

    def save_calibration(self, video_id: int, corners: list[list[float]], zones: dict[str, Any] | None = None) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO calibrations(video_id,corners_json,zones_json) VALUES (?,?,?)",
            (video_id, json.dumps(corners), json.dumps(zones or {})),
        )
        self.connection.commit()

    def add_robot(self, video_id: int, team_number: int, alliance: str, seed_x: int, seed_y: int) -> int:
        cursor = self.connection.execute(
            "INSERT INTO robots(video_id,team_number,alliance,seed_x,seed_y) VALUES (?,?,?,?,?)",
            (video_id, team_number, alliance, seed_x, seed_y),
        )
        self.connection.commit()
        return _inserted_id(cursor)

    def replace_tracks(self, rows: list[tuple[int, int, float, float, float, float]], clear_robot_ids: list[int] | None = None) -> None:
        with self.connection:
            robot_ids = sorted(set(clear_robot_ids or ()) | {row[0] for row in rows})
            if robot_ids:
                placeholders = ",".join("?" for _ in robot_ids)
                self.connection.execute(f"DELETE FROM tracks WHERE robot_id IN ({placeholders})", robot_ids)
            self.connection.executemany(
                "INSERT OR REPLACE INTO tracks(robot_id,frame,t_s,x_ft,y_ft,conf) VALUES (?,?,?,?,?,?)", rows
            )

    def robots_for_video(self, video_id: int):
        return self.connection.execute("SELECT * FROM robots WHERE video_id=? ORDER BY id", (video_id,)).fetchall()

    def tracks_for_video(self, video_id: int):
        return self.connection.execute(
            "SELECT t.robot_id,t.frame,t.t_s,t.x_ft,t.y_ft,t.conf FROM tracks t JOIN robots r ON r.id=t.robot_id WHERE r.video_id=? ORDER BY t.frame,r.id",
            (video_id,),
        ).fetchall()

    def cycles_for_video(self, video_id: int):
        return self.connection.execute(
            "SELECT c.*,r.team_number,r.alliance FROM cycles c JOIN robots r ON r.id=c.robot_id WHERE r.video_id=? ORDER BY c.start_t_s",
            (video_id,),
        ).fetchall()

    def events_for_video(self, video_id: int):
        return self.connection.execute(
            "SELECT e.*,r.team_number,r.alliance,COALESCE(e.scoring_alliance,r.alliance) AS event_alliance FROM scoring_events e LEFT JOIN robots r ON r.id=e.robot_id WHERE e.video_id=? OR r.video_id=? ORDER BY e.t_s",
            (video_id, video_id),
        ).fetchall()

    def replace_analysis(self, robot_ids: list[int], cycles: list[dict], events: list[dict]) -> list[int]:
        if not robot_ids:
            if cycles or events:
                raise ValueError("Analysis results need at least one target robot")
            return []
        event_ids = []
        with self.connection:
            placeholders = ",".join("?" for _ in robot_ids)
            rows = self.connection.execute(
                f"SELECT id,video_id FROM robots WHERE id IN ({placeholders})", robot_ids).fetchall()
            target_ids = set(robot_ids)
            video_ids = {row["video_id"] for row in rows}
            if len(rows) != len(target_ids) or len(video_ids) != 1:
                raise ValueError("Analysis target robots must exist in one video")
            if any(int(c["robot_id"]) not in target_ids for c in cycles):
                raise ValueError("A cycle belongs to a robot outside this analysis")
            if any(event.get("robot_id") is not None and int(event["robot_id"]) not in target_ids for event in events):
                raise ValueError("A generated event belongs to a robot outside this analysis")
            unsupported = {event.get("source", "zone_crossing") for event in events} - GENERATED_SOURCES
            if unsupported:
                raise ValueError(f"Unsupported generated event source: {sorted(unsupported)}")
            video_id = next(iter(video_ids))
            self.connection.execute(
                f"DELETE FROM scoring_events WHERE robot_id IN ({placeholders}) "
                "AND source IN ('zone_crossing','goal_roi') AND status='flagged' "
                "AND NOT EXISTS (SELECT 1 FROM clips WHERE clips.scoring_event_id=scoring_events.id)",
                robot_ids,
            )
            self.connection.execute(
                "DELETE FROM scoring_events WHERE video_id=? AND origin_robot_id=0 "
                "AND source IN ('zone_crossing','goal_roi') AND status='flagged' "
                "AND NOT EXISTS (SELECT 1 FROM clips WHERE clips.scoring_event_id=scoring_events.id)",
                (video_id,),
            )
            self.connection.execute(f"DELETE FROM cycles WHERE robot_id IN ({placeholders})", robot_ids)
            self.connection.executemany(
                "INSERT INTO cycles(robot_id,start_t_s,end_t_s,pickup_zone,scoring_zone,confidence) VALUES (?,?,?,?,?,?)",
                [(c["robot_id"],c["start_t_s"],c["end_t_s"],c["pickup_zone"],c["scoring_zone"],c.get("confidence","medium")) for c in cycles],
            )
            preserved = self.connection.execute(
                "SELECT id,origin_robot_id,source,t_s FROM scoring_events "
                "WHERE video_id=? AND origin_robot_id IS NOT NULL", (video_id,),
            ).fetchall()
            matched = _match_preserved_events(events, preserved)
            # Legacy reviewed or clipped events predate origin_robot_id. Match
            # them by their credited robot with the same one-to-one 0.5 s rule.
            unmatched = [index for index in range(len(events))
                         if index not in matched and events[index].get("robot_id") is not None]
            legacy_rows = [{"id": int(row["id"]), "origin_robot_id": row["robot_id"],
                            "source": row["source"], "t_s": row["t_s"]}
                           for row in self.connection.execute(
                               "SELECT id,robot_id,source,t_s FROM scoring_events WHERE video_id=? "
                               "AND origin_robot_id IS NULL AND robot_id IS NOT NULL "
                               "AND (status IN ('confirmed','rejected') OR EXISTS "
                               "(SELECT 1 FROM clips WHERE clips.scoring_event_id=scoring_events.id))",
                               (video_id,))]
            legacy_matched = _match_preserved_events([events[index] for index in unmatched], legacy_rows)
            for local_index, row_id in legacy_matched.items():
                index = unmatched[local_index]
                matched[index] = row_id
                self.connection.execute("UPDATE scoring_events SET origin_robot_id=? WHERE id=?",
                                        (events[index]["robot_id"], row_id))
            used_retained_ids: set[int] = set(matched.values())
            for index, event in enumerate(events):
                source = event.get("source", "zone_crossing")
                candidate_robot_id = event.get("robot_id")
                origin_robot_id = candidate_robot_id if candidate_robot_id is not None else 0
                retained_id = matched.get(index)
                retained = {"id": retained_id} if retained_id is not None else None
                if not retained and candidate_robot_id is not None:
                    # A legacy reviewed event reassigned to another robot keeps
                    # only an exact, unique timestamp match.
                    legacy = self.connection.execute(
                        "SELECT id,robot_id FROM scoring_events WHERE video_id=? AND origin_robot_id IS NULL "
                        "AND source=? AND status IN ('confirmed','rejected') AND ABS(t_s-?)<0.001",
                        (video_id, source, event["t_s"]),
                    ).fetchall()
                    legacy = [row for row in legacy if row["id"] not in used_retained_ids]
                    same_robot = [row for row in legacy if row["robot_id"] == candidate_robot_id]
                    total_reviewed = self.connection.execute(
                        "SELECT COUNT(*) FROM scoring_events WHERE video_id=? AND source=? "
                        "AND status IN ('confirmed','rejected') AND ABS(t_s-?)<0.001",
                        (video_id, source, event["t_s"]),
                    ).fetchone()[0]
                    retained = (same_robot[0] if len(same_robot) == 1 else
                                legacy[0] if len(legacy) == 1 and total_reviewed == 1 else None)
                    if retained:
                        self.connection.execute("UPDATE scoring_events SET origin_robot_id=? WHERE id=?",
                                                (candidate_robot_id, retained["id"]))
                if retained:
                    used_retained_ids.add(retained["id"])
                    continue
                cursor = self.connection.execute(
                    "INSERT INTO scoring_events(robot_id,video_id,origin_robot_id,t_s,points,confidence,status,source,event_type) VALUES (?,?,?,?,?,?,?,?,?)",
                    (candidate_robot_id, video_id, origin_robot_id, event["t_s"],event.get("points",0),event.get("confidence","low"),event.get("status","flagged"),source,event.get("event_type","unclassified")),
                )
                event_ids.append(_inserted_id(cursor))
        return event_ids

    def add_scoring_event(self, robot_id: int, t_s: float, source: str = "manual") -> int:
        robot = self.connection.execute("SELECT video_id FROM robots WHERE id=?", (robot_id,)).fetchone()
        if robot is None:
            raise ValueError(f"Unknown robot {robot_id}")
        cursor = self.connection.execute(
            "INSERT INTO scoring_events(robot_id,video_id,t_s,points,confidence,status,source,event_type) VALUES (?,?,?,0,'low','flagged',?,'unclassified')",
            (robot_id, robot["video_id"], t_s, source),
        )
        self.connection.commit()
        return _inserted_id(cursor)

    def review_event(self, event_id: int, status: str, points: int, robot_id: int | None | object = _UNCHANGED,
                     event_type: str = _UNCHANGED, confidence: str = _UNCHANGED,
                     scoring_alliance: str | None = _UNCHANGED) -> None:
        """Review an event. Fields left unspecified keep their stored values."""
        if status not in EVENT_STATUSES:
            raise ValueError(f"Unsupported event status: {status}")
        if confidence is not _UNCHANGED and confidence not in CONFIDENCES:
            raise ValueError(f"Unsupported confidence: {confidence}")
        if scoring_alliance is not _UNCHANGED and scoring_alliance is not None and scoring_alliance not in ALLIANCES:
            raise ValueError(f"Unsupported scoring alliance: {scoring_alliance}")
        with self.connection:
            event = self.connection.execute("SELECT video_id,robot_id,event_type,confidence,scoring_alliance FROM scoring_events WHERE id=?", (event_id,)).fetchone()
            if event is None:
                raise ValueError(f"Unknown scoring event {event_id}")
            video_id = event["video_id"]
            if robot_id is _UNCHANGED:
                robot_id = event["robot_id"]
            if event_type is _UNCHANGED:
                event_type = event["event_type"]
            if confidence is _UNCHANGED:
                confidence = event["confidence"]
            if scoring_alliance is _UNCHANGED:
                scoring_alliance = event["scoring_alliance"]
            if robot_id is not None:
                robot = self.connection.execute("SELECT video_id FROM robots WHERE id=?", (robot_id,)).fetchone()
                if robot is None:
                    raise ValueError(f"Unknown robot {robot_id}")
                if video_id is not None and robot["video_id"] != video_id:
                    raise ValueError("Scoring event cannot be assigned to a robot from another video")
                video_id = robot["video_id"]
            # Keep origin_robot_id as the generator identity. Reassignment changes
            # the credited robot, not which candidate was reviewed; another robot
            # may produce independent evidence at the same timestamp.
            self.connection.execute("UPDATE scoring_events SET status=?,points=?,robot_id=?,video_id=?,event_type=?,confidence=?,scoring_alliance=? WHERE id=?",
                                    (status, points, robot_id, video_id, event_type, confidence, scoring_alliance, event_id))

    def set_official_scores(self, video_id: int, red: int | None, blue: int | None,
                            disqualified: set[str] | None = None, swarm_threshold: int = 16,
                            pollinator_1_threshold: int = 4, pollinator_2_threshold: int = 7) -> None:
        disqualified = disqualified or set()
        self.connection.execute("UPDATE videos SET official_red=?,official_blue=?,red_disqualified=?,blue_disqualified=?,swarm_threshold=?,pollinator_1_threshold=?,pollinator_2_threshold=? WHERE id=?",
                                (red, blue, int("red" in disqualified), int("blue" in disqualified), swarm_threshold, pollinator_1_threshold, pollinator_2_threshold, video_id))
        self.connection.commit()

    def set_scoreboard_ocr(self, video_id: int, result: dict) -> None:
        self.connection.execute("UPDATE videos SET scoreboard_ocr_json=? WHERE id=?",
                                (json.dumps(result), video_id))
        self.connection.commit()

    def set_tracking_mode(self, video_id: int, mode: str) -> None:
        if mode not in {"yolo", "seeded_optical_flow"}:
            raise ValueError(f"Unsupported tracking mode: {mode}")
        self.connection.execute("UPDATE videos SET tracking_mode=? WHERE id=?", (mode, video_id))
        self.connection.commit()

    def set_analysis_status(self, video_id: int, status: str) -> None:
        if status not in {"unprocessed", "ready", "running", "failed"}:
            raise ValueError(f"Unsupported analysis status: {status}")
        self.connection.execute("UPDATE videos SET analysis_status=? WHERE id=?", (status, video_id))
        self.connection.commit()

    def add_clip(self, event_id: int, path: Path, start_t_s: float, end_t_s: float) -> None:
        self.connection.execute("INSERT INTO clips(scoring_event_id,path,start_t_s,end_t_s) VALUES (?,?,?,?)",
                                (event_id, str(path), start_t_s, end_t_s))
        self.connection.commit()

    def clips_for_video(self, video_id: int):
        return self.connection.execute(
            "SELECT c.*,e.t_s,e.robot_id,e.points,e.status,e.confidence,r.team_number FROM clips c JOIN scoring_events e ON e.id=c.scoring_event_id LEFT JOIN robots r ON r.id=e.robot_id WHERE e.video_id=? ORDER BY e.t_s",
            (video_id,),
        ).fetchall()

    def save_photos(self, photos: list[tuple[int, Path, float]]) -> None:
        with self.connection:
            for robot_id, path, t_s in photos:
                self.connection.execute("INSERT INTO photos(robot_id,path,t_s) VALUES (?,?,?)", (robot_id, str(path), t_s))

    def replace_photos(self, robot_ids: list[int], photos: list[tuple[int, Path, float]]) -> None:
        with self.connection:
            if robot_ids:
                placeholders = ",".join("?" for _ in robot_ids)
                self.connection.execute(f"DELETE FROM photos WHERE robot_id IN ({placeholders})", robot_ids)
            self.connection.executemany("INSERT INTO photos(robot_id,path,t_s) VALUES (?,?,?)",
                                        [(robot_id, str(path), t_s) for robot_id, path, t_s in photos])

    def photos_for_video(self, video_id: int):
        return self.connection.execute(
            "SELECT p.*,r.team_number FROM photos p JOIN robots r ON r.id=p.robot_id WHERE r.video_id=? ORDER BY r.team_number",
            (video_id,),
        ).fetchall()

    def close(self) -> None:
        self.connection.close()
