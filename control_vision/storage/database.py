from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS videos (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id), path TEXT NOT NULL, duration_s REAL, fps REAL, width INTEGER, height INTEGER, event_code TEXT, match_number TEXT, official_red INTEGER, official_blue INTEGER, red_disqualified INTEGER NOT NULL DEFAULT 0, blue_disqualified INTEGER NOT NULL DEFAULT 0, swarm_threshold INTEGER NOT NULL DEFAULT 16, pollinator_1_threshold INTEGER NOT NULL DEFAULT 4, pollinator_2_threshold INTEGER NOT NULL DEFAULT 7, scoreboard_ocr_json TEXT, tracking_mode TEXT NOT NULL DEFAULT 'yolo');
CREATE TABLE IF NOT EXISTS calibrations (video_id INTEGER PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE, corners_json TEXT NOT NULL, zones_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS robots (id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE, team_number INTEGER NOT NULL, alliance TEXT NOT NULL CHECK(alliance IN ('red','blue')), color_hint TEXT, seed_x INTEGER, seed_y INTEGER);
CREATE TABLE IF NOT EXISTS tracks (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, frame INTEGER NOT NULL, t_s REAL NOT NULL, x_ft REAL NOT NULL, y_ft REAL NOT NULL, conf REAL NOT NULL, UNIQUE(robot_id, frame));
CREATE TABLE IF NOT EXISTS cycles (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, start_t_s REAL NOT NULL, end_t_s REAL NOT NULL, pickup_zone TEXT, scoring_zone TEXT);
CREATE TABLE IF NOT EXISTS scoring_events (id INTEGER PRIMARY KEY, robot_id INTEGER REFERENCES robots(id) ON DELETE SET NULL, video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE, t_s REAL NOT NULL, points INTEGER NOT NULL DEFAULT 0, confidence TEXT NOT NULL DEFAULT 'low', status TEXT NOT NULL DEFAULT 'flagged', source TEXT NOT NULL DEFAULT 'zone_crossing', event_type TEXT NOT NULL DEFAULT 'unclassified', scoring_alliance TEXT CHECK(scoring_alliance IN ('red','blue')));
CREATE TABLE IF NOT EXISTS clips (id INTEGER PRIMARY KEY, scoring_event_id INTEGER REFERENCES scoring_events(id) ON DELETE CASCADE, path TEXT NOT NULL, start_t_s REAL NOT NULL, end_t_s REAL NOT NULL);
CREATE TABLE IF NOT EXISTS photos (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, path TEXT NOT NULL, t_s REAL NOT NULL);
"""


class ProjectDatabase:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        video_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(videos)")}
        for column in ("official_red", "official_blue", "red_disqualified", "blue_disqualified",
                       "swarm_threshold", "pollinator_1_threshold", "pollinator_2_threshold", "scoreboard_ocr_json", "tracking_mode"):
            if column not in video_columns:
                if column == "scoreboard_ocr_json":
                    self.connection.execute("ALTER TABLE videos ADD COLUMN scoreboard_ocr_json TEXT")
                elif column == "tracking_mode":
                    self.connection.execute("ALTER TABLE videos ADD COLUMN tracking_mode TEXT NOT NULL DEFAULT 'yolo'")
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
            self.connection.execute("UPDATE scoring_events SET video_id=(SELECT video_id FROM robots WHERE robots.id=scoring_events.robot_id) WHERE robot_id IS NOT NULL")
        self.connection.commit()

    def add_project(self, name: str) -> int:
        cursor = self.connection.execute("INSERT INTO projects(name) VALUES (?)", (name,))
        self.connection.commit()
        return int(cursor.lastrowid)

    def add_video(self, project_id: int, path: Path, metadata: dict[str, Any], event_code: str | None = None, match_number: str | None = None) -> int:
        cursor = self.connection.execute(
            "INSERT INTO videos(project_id,path,duration_s,fps,width,height,event_code,match_number) VALUES (?,?,?,?,?,?,?,?)",
            (project_id, str(path), metadata["duration_s"], metadata["fps"], metadata["width"], metadata["height"], event_code, match_number),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

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
        return int(cursor.lastrowid)

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
        event_ids = []
        with self.connection:
            if robot_ids:
                placeholders = ",".join("?" for _ in robot_ids)
                self.connection.execute(
                    f"DELETE FROM scoring_events WHERE robot_id IN ({placeholders}) "
                    "AND source IN ('zone_crossing','goal_roi') AND status='flagged' "
                    "AND NOT EXISTS (SELECT 1 FROM clips WHERE clips.scoring_event_id=scoring_events.id)",
                    robot_ids,
                )
                self.connection.execute(f"DELETE FROM cycles WHERE robot_id IN ({placeholders})", robot_ids)
            self.connection.executemany(
                "INSERT INTO cycles(robot_id,start_t_s,end_t_s,pickup_zone,scoring_zone,confidence) VALUES (?,?,?,?,?,?)",
                [(c["robot_id"],c["start_t_s"],c["end_t_s"],c["pickup_zone"],c["scoring_zone"],c.get("confidence","medium")) for c in cycles],
            )
            for event in events:
                source = event.get("source", "zone_crossing")
                retained = self.connection.execute(
                    "SELECT id FROM scoring_events WHERE robot_id=? AND source=? AND ABS(t_s-?)<=0.5 LIMIT 1",
                    (event["robot_id"], source, event["t_s"]),
                ).fetchone()
                if retained:
                    continue
                cursor = self.connection.execute(
                    "INSERT INTO scoring_events(robot_id,video_id,t_s,points,confidence,status,source,event_type) VALUES (?,?,?,?,?,?,?,?)",
                    (event["robot_id"], self.connection.execute("SELECT video_id FROM robots WHERE id=?", (event["robot_id"],)).fetchone()[0], event["t_s"],event.get("points",0),event.get("confidence","low"),event.get("status","flagged"),source,event.get("event_type","unclassified")),
                )
                event_ids.append(int(cursor.lastrowid))
        return event_ids

    def add_scoring_event(self, robot_id: int, t_s: float, source: str = "manual") -> int:
        cursor = self.connection.execute(
            "INSERT INTO scoring_events(robot_id,video_id,t_s,points,confidence,status,source,event_type) VALUES (?,?,?,0,'low','flagged',?,'unclassified')",
            (robot_id, self.connection.execute("SELECT video_id FROM robots WHERE id=?", (robot_id,)).fetchone()[0] if robot_id is not None else None, t_s, source),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def review_event(self, event_id: int, status: str, points: int, robot_id: int | None = None, event_type: str = "unclassified", confidence: str = "low", scoring_alliance: str | None = None) -> None:
        self.connection.execute("UPDATE scoring_events SET status=?,points=?,robot_id=?,event_type=?,confidence=?,scoring_alliance=? WHERE id=?",
                                (status, points, robot_id, event_type, confidence, scoring_alliance, event_id))
        self.connection.commit()

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
