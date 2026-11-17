from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS videos (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id), path TEXT NOT NULL, duration_s REAL, fps REAL, width INTEGER, height INTEGER, event_code TEXT, match_number TEXT, official_red INTEGER, official_blue INTEGER, red_disqualified INTEGER NOT NULL DEFAULT 0, blue_disqualified INTEGER NOT NULL DEFAULT 0, swarm_threshold INTEGER NOT NULL DEFAULT 16, pollinator_1_threshold INTEGER NOT NULL DEFAULT 4, pollinator_2_threshold INTEGER NOT NULL DEFAULT 7, scoreboard_ocr_json TEXT, tracking_mode TEXT NOT NULL DEFAULT 'yolo', analysis_status TEXT NOT NULL DEFAULT 'unprocessed');
CREATE TABLE IF NOT EXISTS calibrations (video_id INTEGER PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE, corners_json TEXT NOT NULL, zones_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS robots (id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE, team_number INTEGER NOT NULL, alliance TEXT NOT NULL CHECK(alliance IN ('red','blue')), color_hint TEXT, seed_x INTEGER, seed_y INTEGER, qualification_disqualified INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS tracks (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, frame INTEGER NOT NULL, t_s REAL NOT NULL, x_ft REAL NOT NULL, y_ft REAL NOT NULL, conf REAL NOT NULL, UNIQUE(robot_id, frame));
CREATE TABLE IF NOT EXISTS cycles (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, start_t_s REAL NOT NULL, end_t_s REAL NOT NULL, pickup_zone TEXT, scoring_zone TEXT);
CREATE TABLE IF NOT EXISTS scoring_events (id INTEGER PRIMARY KEY, robot_id INTEGER REFERENCES robots(id) ON DELETE SET NULL, video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE, origin_robot_id INTEGER, t_s REAL NOT NULL, points INTEGER NOT NULL DEFAULT 0, confidence TEXT NOT NULL DEFAULT 'low', status TEXT NOT NULL DEFAULT 'flagged', source TEXT NOT NULL DEFAULT 'zone_crossing', event_type TEXT NOT NULL DEFAULT 'unclassified', scoring_alliance TEXT CHECK(scoring_alliance IN ('red','blue')));
CREATE TABLE IF NOT EXISTS clips (id INTEGER PRIMARY KEY, scoring_event_id INTEGER REFERENCES scoring_events(id) ON DELETE CASCADE, path TEXT NOT NULL, start_t_s REAL NOT NULL, end_t_s REAL NOT NULL);
CREATE TABLE IF NOT EXISTS photos (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, path TEXT NOT NULL, t_s REAL NOT NULL);
CREATE TABLE IF NOT EXISTS robot_reseeds (id INTEGER PRIMARY KEY, robot_id INTEGER NOT NULL REFERENCES robots(id) ON DELETE CASCADE, t_s REAL NOT NULL, x_px REAL NOT NULL, y_px REAL NOT NULL);
CREATE TABLE IF NOT EXISTS processing_jobs (id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE, position INTEGER NOT NULL, sample_fps INTEGER NOT NULL DEFAULT 8, tracking_mode TEXT NOT NULL, model_path TEXT, status TEXT NOT NULL DEFAULT 'queued', error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""

_UNCHANGED: Any = object()
GENERATED_SOURCES = frozenset({"zone_crossing", "goal_roi"})
EVENT_STATUSES = frozenset({"confirmed", "flagged", "rejected"})
CONFIDENCES = frozenset({"high", "medium", "low"})
ALLIANCES = frozenset({"red", "blue"})
# Events proposed from the live score overlay (control_vision.pipeline.live_score).
LIVE_OVERLAY_SOURCE = "live_overlay"
# A re-read proposal this close to a kept live-overlay event is the same event.
LIVE_OVERLAY_MATCH_S = 1.0
MATCH_TYPES = frozenset({"qualification", "playoff"})
# Confirmed events of these types count toward Ranking Points even at 0 points.
RANKING_COUNT_EVENT_TYPES = frozenset({"hive_tip_auto", "hive_tip_teleop"})


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
        video_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(videos)")}
        added_match_type = "match_type" not in video_columns
        for column, definition in (("match_type", "TEXT NOT NULL DEFAULT 'qualification'"),
                                   ("playoff_first_disqualified", "TEXT"),
                                   ("playoff_simultaneous_dq", "INTEGER NOT NULL DEFAULT 0"),
                                   ("playoff_match_final", "INTEGER NOT NULL DEFAULT 0")):
            if column not in video_columns:
                self.connection.execute(f"ALTER TABLE videos ADD COLUMN {column} {definition}")
        if added_match_type:
            # The legacy alliance-DQ checkboxes were labelled playoff-only.
            # Keep a saved alliance DQ meaningful as a playoff ruling instead
            # of silently applying it to a qualification match.
            self.connection.execute(
                "UPDATE videos SET match_type='playoff' WHERE red_disqualified<>0 OR blue_disqualified<>0")
        robot_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(robots)")}
        for column in ("seed_x", "seed_y"):
            if column not in robot_columns:
                self.connection.execute(f"ALTER TABLE robots ADD COLUMN {column} INTEGER")
        if "qualification_disqualified" not in robot_columns:
            self.connection.execute(
                "ALTER TABLE robots ADD COLUMN qualification_disqualified INTEGER NOT NULL DEFAULT 0")
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
        self._migrate_details()
        if added_analysis_status:
            # Old projects have no processing marker. Preserve access to videos
            # with saved evidence while leaving empty imports unprocessed.
            self.connection.execute(
                "UPDATE videos SET analysis_status='ready' WHERE "
                "EXISTS (SELECT 1 FROM tracks t JOIN robots r ON r.id=t.robot_id WHERE r.video_id=videos.id) "
                "OR EXISTS (SELECT 1 FROM cycles c JOIN robots r ON r.id=c.robot_id WHERE r.video_id=videos.id) "
                "OR EXISTS (SELECT 1 FROM scoring_events e WHERE e.video_id=videos.id)"
            )

    # Project and video fields used by the project library, Home and the New
    # Project wizard. They describe the project; they never change analysis.
    PROJECT_DETAIL_COLUMNS = {
        "event_name": "TEXT", "event_code": "TEXT", "event_date": "TEXT",
        "season": "TEXT NOT NULL DEFAULT 'BIOBUZZ 2026-27'", "notes": "TEXT", "last_opened": "TEXT",
        "model_name": "TEXT", "model_path": "TEXT", "archived": "INTEGER NOT NULL DEFAULT 0",
        "last_exported": "TEXT",
    }
    VIDEO_DETAIL_COLUMNS = {
        "is_holdout": "INTEGER NOT NULL DEFAULT 0", "camera_view": "TEXT NOT NULL DEFAULT 'unknown'",
        "file_size": "INTEGER",
    }
    # A robot marked absent has no seed click; processing only follows seeded robots.
    ROBOT_DETAIL_COLUMNS = {
        "start_position": "TEXT",
        "absent": "INTEGER NOT NULL DEFAULT 0",
    }
    START_POSITIONS = ("left", "right")

    def _migrate_details(self) -> None:
        for table, columns in (("projects", self.PROJECT_DETAIL_COLUMNS), ("videos", self.VIDEO_DETAIL_COLUMNS),
                               ("robots", self.ROBOT_DETAIL_COLUMNS)):
            existing = {row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})")}
            for column, definition in columns.items():
                if column not in existing:
                    self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def project_details(self, project_id: int) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        return dict(row) if row else None

    def set_project_details(self, project_id: int, **fields: Any) -> None:
        """Update descriptive project fields; unknown field names are refused."""
        unknown = set(fields) - set(self.PROJECT_DETAIL_COLUMNS) - {"name"}
        if unknown:
            raise ValueError("Unknown project field(s): " + ", ".join(sorted(unknown)))
        if "event_code" in fields and fields["event_code"] is not None:
            fields["event_code"] = normalise_event_code(fields["event_code"])
        if not fields:
            return
        assignments = ",".join(f"{column}=?" for column in fields)
        with self.connection:
            self.connection.execute(f"UPDATE projects SET {assignments} WHERE id=?", (*fields.values(), project_id))

    def touch_project(self, project_id: int, when: str) -> None:
        """Record when the project was last opened (an ISO timestamp)."""
        with self.connection:
            self.connection.execute("UPDATE projects SET last_opened=? WHERE id=?", (when, project_id))

    def set_video_details(self, video_id: int, *, is_holdout: bool | None = None, camera_view: str | None = None,
                          file_size: int | None = None) -> None:
        if camera_view is not None and camera_view not in {"wide", "close", "unknown"}:
            raise ValueError(f"Unknown camera view: {camera_view}")
        fields = {key: value for key, value in (("is_holdout", None if is_holdout is None else int(bool(is_holdout))),
                                                ("camera_view", camera_view), ("file_size", file_size)) if value is not None}
        if not fields:
            return
        assignments = ",".join(f"{column}=?" for column in fields)
        with self.connection:
            self.connection.execute(f"UPDATE videos SET {assignments} WHERE id=?", (*fields.values(), video_id))

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

    def add_reseed(self, video_id: int, robot_id: int, t_s: float, x_px: float, y_px: float) -> int:
        """Store a reviewer's mid-match click: robot ``robot_id`` is at (x_px, y_px) in source pixels at ``t_s``.

        Applied on the next analysis run (``track_video(reseeds=...)``)."""
        values = (float(t_s), float(x_px), float(y_px))
        if not all(math.isfinite(value) for value in values) or values[0] < 0:
            raise ValueError("A re-seed needs a finite, non-negative time and a finite position")
        robot = self.connection.execute("SELECT 1 FROM robots WHERE id=? AND video_id=?", (robot_id, video_id)).fetchone()
        if robot is None:
            raise ValueError(f"Robot {robot_id} does not belong to video {video_id}")
        cursor = self.connection.execute("INSERT INTO robot_reseeds(robot_id,t_s,x_px,y_px) VALUES (?,?,?,?)",
                                         (robot_id, *values))
        self.connection.commit()
        return _inserted_id(cursor)

    def delete_reseed(self, reseed_id: int) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM robot_reseeds WHERE id=?", (reseed_id,))

    def reseeds_for_video(self, video_id: int):
        """Rows (id, robot_id, t_s, x_px, y_px) in time order."""
        return self.connection.execute(
            "SELECT s.id,s.robot_id,s.t_s,s.x_px,s.y_px FROM robot_reseeds s JOIN robots r ON r.id=s.robot_id "
            "WHERE r.video_id=? ORDER BY s.t_s,s.id", (video_id,)).fetchall()

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

    def calibration_for(self, video_id: int) -> tuple[list[list[float]], dict[str, Any]] | None:
        """Saved corners (video pixels) and zones for a video, or None."""
        row = self.connection.execute("SELECT corners_json,zones_json FROM calibrations WHERE video_id=?",
                                      (video_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row["corners_json"]), json.loads(row["zones_json"] or "{}")

    def save_identity(self, video_id: int, entries: list[dict[str, Any]], mark_unprocessed: bool = False) -> list[int]:
        """Save a match's robot identities in one transaction; returns the robot ids in entry order.

        Each entry has team_number, alliance, start_position (or None), absent,
        seed_x / seed_y (None when absent) and, for a robot already saved, its
        id. Existing robots are updated in place so tracks, cycles and events
        that point at them keep pointing at the right robot. Robots of this
        video that are not in ``entries`` are deleted with their results.
        With ``mark_unprocessed`` the video's results are marked out of date in
        the same transaction, so robots and status can never disagree.
        """
        teams = [int(entry["team_number"]) for entry in entries]
        if len(teams) != len(set(teams)):
            raise ValueError("A team is assigned twice in this match.")
        existing = {int(row["id"]) for row in self.robots_for_video(video_id)}
        ids = []
        with self.connection:
            for entry in entries:
                alliance = entry["alliance"]
                if alliance not in ("red", "blue"):
                    raise ValueError(f"Unknown alliance: {alliance}")
                position = entry.get("start_position")
                if position is not None and position not in self.START_POSITIONS:
                    raise ValueError(f"Unknown starting position: {position}")
                absent = bool(entry.get("absent"))
                seed_x = None if absent else entry.get("seed_x")
                seed_y = None if absent else entry.get("seed_y")
                if not absent and (seed_x is None or seed_y is None):
                    raise ValueError(f"Team {entry['team_number']} needs a click on the robot, or mark it not visible.")
                values = (int(entry["team_number"]), alliance, None if seed_x is None else int(seed_x),
                          None if seed_y is None else int(seed_y), position, int(absent))
                robot_id = entry.get("id")
                if robot_id is not None and int(robot_id) in existing:
                    self.connection.execute(
                        "UPDATE robots SET team_number=?,alliance=?,seed_x=?,seed_y=?,start_position=?,absent=? WHERE id=?",
                        (*values, int(robot_id)))
                    ids.append(int(robot_id))
                else:
                    cursor = self.connection.execute(
                        "INSERT INTO robots(video_id,team_number,alliance,seed_x,seed_y,start_position,absent) "
                        "VALUES (?,?,?,?,?,?,?)", (video_id, *values))
                    ids.append(_inserted_id(cursor))
            for robot_id in existing - set(ids):
                self.connection.execute("DELETE FROM robots WHERE id=?", (robot_id,))
            if mark_unprocessed:
                self.connection.execute("UPDATE videos SET analysis_status='unprocessed' WHERE id=?", (video_id,))
        return ids

    # ----- processing queue -----

    JOB_STATUSES = ("queued", "running", "done", "failed", "cancelled")

    def queue_jobs(self, video_ids, sample_fps: int, tracking_mode: str, model_path: str | None) -> list[int]:
        """Queue videos for processing with the settings each job will use; a video already waiting is skipped."""
        if not 5 <= int(sample_fps) <= 10:
            raise ValueError("The sample rate must be 5 to 10 frames per second.")
        if tracking_mode not in ("yolo", "seeded_optical_flow"):
            raise ValueError(f"Unknown tracking mode: {tracking_mode}")
        waiting = {int(row[0]) for row in self.connection.execute(
            "SELECT video_id FROM processing_jobs WHERE status IN ('queued','running')")}
        ids = []
        with self.connection:
            position = self.connection.execute("SELECT COALESCE(MAX(position), 0) FROM processing_jobs").fetchone()[0]
            for video_id in video_ids:
                if int(video_id) in waiting:
                    continue
                waiting.add(int(video_id))
                position += 1
                cursor = self.connection.execute(
                    "INSERT INTO processing_jobs(video_id,position,sample_fps,tracking_mode,model_path) VALUES (?,?,?,?,?)",
                    (int(video_id), position, int(sample_fps), tracking_mode, model_path or None))
                ids.append(_inserted_id(cursor))
        return ids

    def processing_jobs(self):
        return self.connection.execute(
            "SELECT j.*, v.path, v.event_code, v.match_number, v.duration_s FROM processing_jobs j "
            "JOIN videos v ON v.id=j.video_id ORDER BY j.position, j.id").fetchall()

    def next_queued_job(self):
        return self.connection.execute(
            "SELECT j.*, v.path FROM processing_jobs j JOIN videos v ON v.id=j.video_id "
            "WHERE j.status='queued' ORDER BY j.position, j.id LIMIT 1").fetchone()

    def set_job_status(self, job_id: int, status: str, error: str | None = None) -> None:
        if status not in self.JOB_STATUSES:
            raise ValueError(f"Unknown job status: {status}")
        with self.connection:
            self.connection.execute("UPDATE processing_jobs SET status=?, error=? WHERE id=?", (status, error, int(job_id)))

    def move_job(self, job_id: int, offset: int) -> bool:
        """Swap a job with its neighbour in the queue order; False at either end."""
        rows = [int(row[0]) for row in self.connection.execute("SELECT id FROM processing_jobs ORDER BY position, id")]
        if int(job_id) not in rows:
            return False
        index = rows.index(int(job_id))
        target = index + int(offset)
        if not 0 <= target < len(rows):
            return False
        rows[index], rows[target] = rows[target], rows[index]
        with self.connection:
            for position, row_id in enumerate(rows, start=1):
                self.connection.execute("UPDATE processing_jobs SET position=? WHERE id=?", (position, row_id))
        return True

    def remove_jobs(self, job_ids) -> None:
        with self.connection:
            for job_id in job_ids:
                self.connection.execute("DELETE FROM processing_jobs WHERE id=? AND status!='running'", (int(job_id),))

    def clear_finished_jobs(self) -> int:
        with self.connection:
            return self.connection.execute(
                "DELETE FROM processing_jobs WHERE status IN ('done','cancelled')").rowcount

    def retry_failed_jobs(self) -> int:
        with self.connection:
            return self.connection.execute(
                "UPDATE processing_jobs SET status='queued', error=NULL WHERE status IN ('failed','cancelled')").rowcount

    def recover_interrupted_jobs(self) -> int:
        """A job left running when the app closed saved nothing; queue it again. Returns how many jobs wait."""
        with self.connection:
            self.connection.execute("UPDATE processing_jobs SET status='queued' WHERE status='running'")
        return self.connection.execute("SELECT COUNT(*) FROM processing_jobs WHERE status='queued'").fetchone()[0]

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
            counts = points != 0 or event_type in RANKING_COUNT_EVENT_TYPES
            if status == "confirmed" and counts and robot_id is None and scoring_alliance is None:
                # The scoreboard credits an alliance from the recipient or the
                # attributed robot. With neither, confirmed points (or a HIVE
                # TIP, which counts toward POLLINATOR Ranking Points whatever
                # its points) would count for no one, so refuse instead of
                # losing them silently.
                raise ValueError("Choose the attributed robot or the alliance that receives these points "
                                 "before confirming them.")
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
        """Save official totals and RP thresholds.

        ``disqualified`` left as None keeps the stored match ruling. A set is
        a playoff alliance DQ and is validated like ``set_match_ruling``.
        """
        with self.connection:
            if disqualified is not None:
                ruling = self.match_ruling(video_id)
                both = set(disqualified) == {"red", "blue"}
                self._write_match_ruling(
                    video_id, ruling["match_type"],
                    qualification_disqualified_robot_ids=ruling["qualification_disqualified_robot_ids"],
                    playoff_disqualified=set(disqualified),
                    first_disqualified=ruling["first_disqualified"] if both else None,
                    simultaneous_disqualification=ruling["simultaneous_disqualification"] and both,
                    match_final=ruling["match_final"])
            self.connection.execute("UPDATE videos SET official_red=?,official_blue=?,swarm_threshold=?,pollinator_1_threshold=?,pollinator_2_threshold=? WHERE id=?",
                                    (red, blue, swarm_threshold, pollinator_1_threshold, pollinator_2_threshold, video_id))

    def match_ruling(self, video_id: int) -> dict[str, Any]:
        """The stored match type and referee ruling for one video."""
        video = self.connection.execute(
            "SELECT match_type,red_disqualified,blue_disqualified,playoff_first_disqualified,"
            "playoff_simultaneous_dq,playoff_match_final FROM videos WHERE id=?", (video_id,)).fetchone()
        if video is None:
            raise ValueError(f"Unknown video {video_id}")
        team_ids = {int(row["id"]) for row in self.connection.execute(
            "SELECT id FROM robots WHERE video_id=? AND qualification_disqualified<>0", (video_id,))}
        return {"match_type": video["match_type"],
                "qualification_disqualified_robot_ids": team_ids,
                "playoff_disqualified": {alliance for alliance in ("red", "blue")
                                         if video[f"{alliance}_disqualified"]},
                "first_disqualified": video["playoff_first_disqualified"],
                "simultaneous_disqualification": bool(video["playoff_simultaneous_dq"]),
                "match_final": bool(video["playoff_match_final"])}

    def set_match_ruling(self, video_id: int, match_type: str, *,
                         qualification_disqualified_robot_ids: Iterable[int] = (),
                         playoff_disqualified: Iterable[str] = (),
                         first_disqualified: str | None = None,
                         simultaneous_disqualification: bool = False,
                         match_final: bool = False) -> None:
        """Save the match type and the reviewer-entered referee ruling atomically.

        Qualification DQ is per team (T601); playoff DQ is per alliance, with
        the Head Referee's chronology or simultaneous ruling for a double DQ
        (T705). Rulings for the other match type are refused, never stored.
        """
        with self.connection:
            self._write_match_ruling(
                video_id, match_type,
                qualification_disqualified_robot_ids=qualification_disqualified_robot_ids,
                playoff_disqualified=playoff_disqualified, first_disqualified=first_disqualified,
                simultaneous_disqualification=simultaneous_disqualification, match_final=match_final)

    def set_match_type(self, video_id: int, match_type: str, *, clear_inapplicable: bool = False) -> None:
        """Change the match type.

        Rulings that only apply to the old type are refused with a plain
        message unless ``clear_inapplicable`` is set, which removes them in
        the same transaction.
        """
        if match_type not in MATCH_TYPES:
            raise ValueError("Match type must be qualification or playoff")
        with self.connection:
            ruling = self.match_ruling(video_id)
            playoff_ruling = bool(ruling["playoff_disqualified"] or ruling["first_disqualified"]
                                  or ruling["simultaneous_disqualification"] or ruling["match_final"])
            team_ruling = bool(ruling["qualification_disqualified_robot_ids"])
            inapplicable = playoff_ruling if match_type == "qualification" else team_ruling
            if match_type == ruling["match_type"]:
                return
            if inapplicable and not clear_inapplicable:
                if match_type == "qualification":
                    raise ValueError("This match has a playoff alliance DQ or ruling. Clear it before "
                                     "marking the match as a qualification.")
                raise ValueError("This match has a qualification team DQ. Clear it before marking "
                                 "the match as a playoff.")
            self._write_match_ruling(
                video_id, match_type,
                qualification_disqualified_robot_ids=(ruling["qualification_disqualified_robot_ids"]
                                                      if match_type == "qualification" else ()),
                playoff_disqualified=ruling["playoff_disqualified"] if match_type == "playoff" else (),
                first_disqualified=ruling["first_disqualified"] if match_type == "playoff" else None,
                simultaneous_disqualification=(ruling["simultaneous_disqualification"]
                                               if match_type == "playoff" else False),
                match_final=ruling["match_final"] if match_type == "playoff" else False)

    def _write_match_ruling(self, video_id: int, match_type: str, *,
                            qualification_disqualified_robot_ids: Iterable[int],
                            playoff_disqualified: Iterable[str], first_disqualified: str | None,
                            simultaneous_disqualification: bool, match_final: bool) -> None:
        """Validate and write a ruling; the caller owns the transaction."""
        if match_type not in MATCH_TYPES:
            raise ValueError("Match type must be qualification or playoff")
        raw_ids = list(qualification_disqualified_robot_ids)
        if any(isinstance(robot_id, bool) or not isinstance(robot_id, int) for robot_id in raw_ids):
            raise ValueError("Qualification DQ robot IDs must be integers")
        team_ids = set(raw_ids)
        alliances = set(playoff_disqualified)
        if alliances - ALLIANCES:
            raise ValueError(f"Unknown playoff DQ alliance: {sorted(alliances - ALLIANCES)}")
        if first_disqualified is not None and first_disqualified not in ALLIANCES:
            raise ValueError(f"Unknown first disqualified alliance: {first_disqualified}")
        if first_disqualified is not None and alliances != ALLIANCES:
            raise ValueError("A first-DQ ruling needs both playoff alliances disqualified")
        if simultaneous_disqualification and (alliances != ALLIANCES or first_disqualified is not None):
            raise ValueError("A simultaneous DQ ruling needs both alliances disqualified and no first DQ")
        if team_ids and alliances:
            raise ValueError("A qualification team DQ cannot be combined with a playoff alliance DQ")
        if match_type == "qualification" and (alliances or first_disqualified is not None
                                              or simultaneous_disqualification or match_final):
            raise ValueError("Alliance DQ, referee DQ rulings and match final apply to playoff matches only")
        if match_type == "playoff" and team_ids:
            raise ValueError("A playoff DQ applies to the whole alliance; team DQ is for qualification matches")
        if self.connection.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is None:
            raise ValueError(f"Unknown video {video_id}")
        video_robot_ids = {int(row["id"]) for row in self.connection.execute(
            "SELECT id FROM robots WHERE video_id=?", (video_id,))}
        if team_ids - video_robot_ids:
            raise ValueError(f"Qualification DQ robots are not in this video: {sorted(team_ids - video_robot_ids)}")
        self.connection.execute(
            "UPDATE videos SET match_type=?,red_disqualified=?,blue_disqualified=?,playoff_first_disqualified=?,"
            "playoff_simultaneous_dq=?,playoff_match_final=? WHERE id=?",
            (match_type, int("red" in alliances), int("blue" in alliances), first_disqualified,
             int(bool(simultaneous_disqualification)), int(bool(match_final)), video_id))
        self.connection.execute("UPDATE robots SET qualification_disqualified=0 WHERE video_id=?", (video_id,))
        self.connection.executemany("UPDATE robots SET qualification_disqualified=1 WHERE video_id=? AND id=?",
                                    [(video_id, robot_id) for robot_id in sorted(team_ids)])

    def replace_live_overlay_events(self, video_id: int, events: list[dict],
                                    read_until_s: float | None = None) -> dict[str, int]:
        """Store live-overlay proposals for a video, replacing earlier unreviewed ones.

        Flagged live-overlay events without a clip are removed; for a read
        that stopped early (``read_until_s`` given) only those up to that time,
        so proposals from an earlier, longer read survive. Events a
        reviewer confirmed or rejected, or that have a clip, are kept, and a
        new proposal for the same alliance within LIVE_OVERLAY_MATCH_S of a
        kept one (each kept event matching once) is not added again, so reading the
        overlay twice neither duplicates nor undoes a review. New events are
        alliance-scoped (no robot) and flagged.
        """
        for event in events:
            if event.get("scoring_alliance") not in ALLIANCES:
                raise ValueError("A live-overlay event needs a red or blue scoring alliance")
            if event.get("confidence", "low") not in CONFIDENCES:
                raise ValueError(f"Unsupported confidence: {event.get('confidence')}")
        with self.connection:
            if self.connection.execute("SELECT 1 FROM videos WHERE id=?", (video_id,)).fetchone() is None:
                raise ValueError(f"Unknown video {video_id}")
            removed = self.connection.execute(
                "DELETE FROM scoring_events WHERE video_id=? AND source=? AND status='flagged' "
                "AND (? IS NULL OR t_s<=?) "
                "AND NOT EXISTS (SELECT 1 FROM clips WHERE clips.scoring_event_id=scoring_events.id)",
                (video_id, LIVE_OVERLAY_SOURCE, read_until_s, read_until_s)).rowcount
            kept = [dict(row) for row in self.connection.execute(
                "SELECT id,t_s,event_type,scoring_alliance FROM scoring_events WHERE video_id=? AND source=?",
                (video_id, LIVE_OVERLAY_SOURCE))]
            used: set[int] = set()
            added = 0
            for event in sorted(events, key=lambda item: float(item["t_s"])):
                # Not matched on event type: the reviewer may have changed it.
                same = next((row for row in kept if row["id"] not in used
                             and row["scoring_alliance"] == event["scoring_alliance"]
                             and abs(float(row["t_s"]) - float(event["t_s"])) <= LIVE_OVERLAY_MATCH_S), None)
                if same is not None:
                    used.add(same["id"])
                    continue
                self.connection.execute(
                    "INSERT INTO scoring_events(robot_id,video_id,t_s,points,confidence,status,source,event_type,"
                    "scoring_alliance) VALUES (NULL,?,?,?,?,'flagged',?,?,?)",
                    (video_id, float(event["t_s"]), int(event.get("points", 0)), event.get("confidence", "low"),
                     LIVE_OVERLAY_SOURCE, event.get("event_type", "unclassified"), event["scoring_alliance"]))
                added += 1
        return {"added": added, "kept": len(kept), "removed": removed}

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


def normalise_event_code(code: str) -> str:
    """Event codes are stored as typed, uppercased, without spaces."""
    return "".join(str(code).split()).upper()
