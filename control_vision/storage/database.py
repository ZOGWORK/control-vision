from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS videos (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id), path TEXT NOT NULL, duration_s REAL, fps REAL, width INTEGER, height INTEGER, event_code TEXT, match_number TEXT);
CREATE TABLE IF NOT EXISTS calibrations (video_id INTEGER PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE, corners_json TEXT NOT NULL, zones_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS robots (id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE, team_number INTEGER NOT NULL, alliance TEXT NOT NULL CHECK(alliance IN ('red','blue')), color_hint TEXT, seed_x INTEGER, seed_y INTEGER);
CREATE TABLE IF NOT EXISTS tracks (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, frame INTEGER NOT NULL, t_s REAL NOT NULL, x_ft REAL NOT NULL, y_ft REAL NOT NULL, conf REAL NOT NULL, UNIQUE(robot_id, frame));
CREATE TABLE IF NOT EXISTS cycles (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, start_t_s REAL NOT NULL, end_t_s REAL NOT NULL, pickup_zone TEXT, scoring_zone TEXT);
CREATE TABLE IF NOT EXISTS scoring_events (id INTEGER PRIMARY KEY, robot_id INTEGER REFERENCES robots(id) ON DELETE SET NULL, t_s REAL NOT NULL, points INTEGER NOT NULL DEFAULT 0, confidence TEXT NOT NULL DEFAULT 'low', status TEXT NOT NULL DEFAULT 'flagged', source TEXT NOT NULL DEFAULT 'zone_crossing');
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
        robot_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(robots)")}
        for column in ("seed_x", "seed_y"):
            if column not in robot_columns:
                self.connection.execute(f"ALTER TABLE robots ADD COLUMN {column} INTEGER")
        self.connection.commit()

    def add_project(self, name: str) -> int:
        cursor = self.connection.execute("INSERT INTO projects(name) VALUES (?)", (name,))
        self.connection.commit()
        return int(cursor.lastrowid)

    def add_video(self, project_id: int, path: Path, metadata: dict[str, Any]) -> int:
        cursor = self.connection.execute(
            "INSERT INTO videos(project_id,path,duration_s,fps,width,height) VALUES (?,?,?,?,?,?)",
            (project_id, str(path), metadata["duration_s"], metadata["fps"], metadata["width"], metadata["height"]),
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

    def close(self) -> None:
        self.connection.close()
