"""The app's own store, outside any project: known projects and model versions.

Each project is one SQLite file chosen by the user. The library only remembers
where those files are (``projects.json``) and reads summaries from them
read-only, so a missing or moved file never crashes a list. Model versions are
registered in ``models.json`` by path; weights are never copied or downloaded.

Nothing here invents a number: a value the data does not hold is ``None`` and
the UI shows "—" or "Not measured".
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

HOME_ENV = "CONTROL_VISION_HOME"
SEASON = "BIOBUZZ 2026-27"
PROJECT_STATUSES = ("Not processed", "Processing", "Ready to review", "Reviewed", "Exported")
MODEL_CLASSES = ("robot", "yellow_pollen", "red_nectar", "blue_nectar")


_TEST_HOME: Path | None = None


def _running_under_unittest() -> bool:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    return bool(spec and spec.name == "unittest.__main__")


def app_home() -> Path:
    """Folder for the app's own files: ``$CONTROL_VISION_HOME`` or the user's app-data folder.

    Under the unittest runner, without an override, it is a temporary folder,
    so tests never add entries to the user's real project or model list.
    """
    global _TEST_HOME
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override)
    if _running_under_unittest():
        if _TEST_HOME is None:
            _TEST_HOME = Path(tempfile.mkdtemp(prefix="control-vision-tests-"))
        return _TEST_HOME
    from PySide6.QtCore import QStandardPaths
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)
    return Path(base or Path.home()) / "Control Vision"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path, default: dict) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else dict(default)
    except (OSError, ValueError):
        return dict(default)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class ProjectSummary:
    path: Path
    missing: bool = False
    unreadable: str | None = None
    name: str = ""
    event_name: str | None = None
    event_code: str | None = None
    event_date: str | None = None
    season: str = SEASON
    notes: str | None = None
    last_opened: str | None = None
    model_name: str | None = None
    archived: bool = False
    matches: int = 0
    processed: int = 0
    running: int = 0
    flagged: int = 0
    cycles: int = 0
    processed_seconds: float = 0.0
    last_exported: str | None = None
    teams: set[int] = field(default_factory=set)
    thumbnail_video: Path | None = None
    holdout: int = 0

    @property
    def status(self) -> str:
        if self.running:
            return "Processing"
        if not self.processed:
            return "Not processed"
        if self.flagged:
            return "Ready to review"
        if self.last_exported:
            return "Exported"
        return "Reviewed"

    @property
    def display_name(self) -> str:
        # Every project file is control_vision.sqlite, so its folder names it best.
        return self.name or self.path.parent.name or self.path.stem


def summarize_project(path: Path) -> ProjectSummary:
    """Read one project file without changing it."""
    path = Path(path)
    summary = ProjectSummary(path=path)
    if not path.is_file():
        summary.missing = True
        return summary
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        summary.unreadable = str(exc)
        return summary
    connection.row_factory = sqlite3.Row
    try:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(projects)")}
        project = connection.execute("SELECT * FROM projects ORDER BY id LIMIT 1").fetchone()
        if project is None:
            summary.unreadable = "This file has no Control Vision project."
            return summary
        for name in ("name", "event_name", "event_code", "event_date", "notes", "last_opened", "model_name",
                     "last_exported"):
            if name in columns:
                setattr(summary, name, project[name])
        if "season" in columns and project["season"]:
            summary.season = project["season"]
        summary.archived = bool("archived" in columns and project["archived"])
        video_columns = {row["name"] for row in connection.execute("PRAGMA table_info(videos)")}
        has_status = "analysis_status" in video_columns
        has_holdout = "is_holdout" in video_columns
        for video in connection.execute("SELECT * FROM videos ORDER BY id"):
            summary.matches += 1
            status = video["analysis_status"] if has_status else "unprocessed"
            if has_holdout and video["is_holdout"]:
                summary.holdout += 1
            if status == "running":
                summary.running += 1
            if status == "ready":
                summary.processed += 1
                summary.processed_seconds += float(video["duration_s"] or 0)
                if summary.thumbnail_video is None:
                    summary.thumbnail_video = Path(video["path"])
        summary.teams = {int(row[0]) for row in connection.execute("SELECT DISTINCT team_number FROM robots")}
        summary.cycles = int(connection.execute(
            "SELECT COUNT(*) FROM cycles c JOIN robots r ON r.id=c.robot_id JOIN videos v ON v.id=r.video_id "
            + ("WHERE v.analysis_status='ready'" if has_status else "")).fetchone()[0])
        event_columns = {row["name"] for row in connection.execute("PRAGMA table_info(scoring_events)")}
        if "video_id" in event_columns and has_status:
            summary.flagged = int(connection.execute(
                "SELECT COUNT(*) FROM scoring_events e JOIN videos v ON v.id=e.video_id "
                "WHERE e.status='flagged' AND v.analysis_status='ready'").fetchone()[0])
    except sqlite3.Error as exc:
        summary.unreadable = str(exc)
    finally:
        connection.close()
    return summary


@dataclass
class HomeStats:
    matches_analysed: int = 0
    cycles: int = 0
    hours_processed: float = 0.0
    projects: int = 0
    ready_to_review: int = 0


def home_stats(summaries: list[ProjectSummary]) -> HomeStats:
    stats = HomeStats()
    for summary in summaries:
        if summary.missing or summary.unreadable or summary.archived:
            continue
        stats.projects += 1
        stats.matches_analysed += summary.processed
        stats.cycles += summary.cycles
        stats.hours_processed += summary.processed_seconds / 3600
        stats.ready_to_review += summary.status == "Ready to review"
    return stats


class ProjectLibrary:
    """Known project files, most recently used first."""

    def __init__(self, home: Path | None = None):
        self.home = Path(home) if home else app_home()
        self.file = self.home / "projects.json"

    def _load(self) -> dict:
        data = _read_json(self.file, {"projects": [], "last": None})
        data.setdefault("projects", [])
        return data

    def paths(self) -> list[Path]:
        return [Path(entry) for entry in self._load()["projects"] if isinstance(entry, str)]

    def last(self) -> Path | None:
        last = self._load().get("last")
        return Path(last) if isinstance(last, str) else None

    def register(self, path: Path, make_last: bool = True) -> None:
        path = Path(path).resolve()
        data = self._load()
        entries = [entry for entry in data["projects"] if Path(entry).resolve() != path]
        data["projects"] = [str(path)] + entries
        if make_last:
            data["last"] = str(path)
        _write_json(self.file, data)

    def forget(self, path: Path) -> None:
        path = Path(path).resolve()
        data = self._load()
        data["projects"] = [entry for entry in data["projects"] if Path(entry).resolve() != path]
        if data.get("last") and Path(data["last"]).resolve() == path:
            data["last"] = None
        _write_json(self.file, data)

    def replace(self, old: Path, new: Path) -> None:
        """Point an entry at a file the user located after it moved."""
        old, new = Path(old).resolve(), Path(new).resolve()
        data = self._load()
        data["projects"] = [str(new) if Path(entry).resolve() == old else entry for entry in data["projects"]]
        if data.get("last") and Path(data["last"]).resolve() == old:
            data["last"] = str(new)
        _write_json(self.file, data)

    def summaries(self) -> list[ProjectSummary]:
        return [summarize_project(path) for path in self.paths()]


def unique_name(name: str, existing: set[str]) -> str:
    """``name``, or ``name (2)``, ``name (3)``… when it is already used."""
    name = name.strip()
    lowered = {item.casefold() for item in existing}
    if name.casefold() not in lowered:
        return name
    number = 2
    while f"{name} ({number})".casefold() in lowered:
        number += 1
    return f"{name} ({number})"


def duplicate_project(source: Path, target: Path, new_name: str) -> Path:
    """Copy a project's videos and calibrations into a new project file, without results.

    Robots, tracks, cycles, events, clips, photos, official scores, OCR
    readings and rulings are not copied. Holdout flags and camera views are.
    """
    from control_vision.storage.database import ProjectDatabase

    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(f"{target} already exists.")
    original = ProjectDatabase(source)
    try:
        project = original.connection.execute("SELECT * FROM projects ORDER BY id LIMIT 1").fetchone()
        if project is None:
            raise ValueError("This file has no Control Vision project.")
        videos = original.connection.execute("SELECT * FROM videos WHERE project_id=? ORDER BY id", (project["id"],)).fetchall()
        calibrations = {row["video_id"]: row for row in original.connection.execute("SELECT * FROM calibrations")}
    finally:
        original.close()
    copy = ProjectDatabase(target)
    try:
        project_id = copy.add_project(new_name)
        copy.set_project_details(project_id, **{key: project[key] for key in
                                                ("event_name", "event_code", "event_date", "season", "notes",
                                                 "model_name", "model_path")})
        for video in videos:
            metadata = {"duration_s": video["duration_s"], "fps": video["fps"], "width": video["width"],
                        "height": video["height"]}
            video_id = copy.add_video(project_id, Path(video["path"]), metadata, video["event_code"], video["match_number"])
            copy.set_video_details(video_id, is_holdout=bool(video["is_holdout"]), camera_view=video["camera_view"],
                                   file_size=video["file_size"])
            calibration = calibrations.get(video["id"])
            if calibration is not None:
                copy.save_calibration(video_id, json.loads(calibration["corners_json"]),
                                      json.loads(calibration["zones_json"] or "{}"))
    finally:
        copy.close()
    return target


def export_project_file(source: Path, target: Path) -> Path:
    """Save a consistent copy of a project file; the original is not moved."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("Choose a different file to export to.")
    reader = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    try:
        writer = sqlite3.connect(target)
        try:
            reader.backup(writer)
        finally:
            writer.close()
    finally:
        reader.close()
    return target


# ----- Model versions -----

@dataclass
class ModelVersion:
    id: str
    name: str
    path: str
    created: str | None = None
    dataset: str | None = None
    base_weights: str | None = None
    epochs: int | None = None
    map50: float | None = None
    map50_95: float | None = None
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    status: str = "Active"
    notes: str | None = None

    @property
    def measured(self) -> bool:
        return self.map50 is not None

    def class_metric(self, class_name: str, metric: str) -> float | None:
        value = self.per_class.get(class_name, {}).get(metric)
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def file_exists(self) -> bool:
        return Path(self.path).is_file()

    @property
    def file_size(self) -> int | None:
        try:
            return Path(self.path).stat().st_size
        except OSError:
            return None


def _metrics_sidecar(model_path: Path) -> dict[str, Any]:
    """Metrics saved next to the weights as ``<name>.metrics.json`` or ``<name>.json``, if any."""
    for candidate in (model_path.with_suffix(".metrics.json"), model_path.with_suffix(".json")):
        if candidate.is_file():
            data = _read_json(candidate, {})
            if data:
                return data
    return {}


class ModelStore:
    """Registered model versions and the active one for new projects."""

    def __init__(self, home: Path | None = None):
        self.home = Path(home) if home else app_home()
        self.file = self.home / "models.json"

    def _load(self) -> dict:
        data = _read_json(self.file, {"models": [], "active": None})
        data.setdefault("models", [])
        return data

    def versions(self) -> list[ModelVersion]:
        versions = []
        for entry in self._load()["models"]:
            try:
                versions.append(ModelVersion(**entry))
            except TypeError:
                continue
        return versions

    def active_id(self) -> str | None:
        return self._load().get("active")

    def active(self) -> ModelVersion | None:
        active = self.active_id()
        return next((model for model in self.versions() if model.id == active), None)

    def get(self, model_id: str) -> ModelVersion | None:
        return next((model for model in self.versions() if model.id == model_id), None)

    def add_file(self, path: Path, name: str | None = None) -> ModelVersion:
        """Register a weights file by path; metrics come only from a sidecar file next to it."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No model file at {path}")
        data = self._load()
        metrics = _metrics_sidecar(path)
        used = {entry.get("id") for entry in data["models"]}
        base_id = (name or path.stem).strip() or "model"
        model_id, number = base_id, 2
        while model_id in used:
            model_id, number = f"{base_id}-{number}", number + 1

        def number_or_none(key):
            value = metrics.get(key)
            return float(value) if isinstance(value, (int, float)) else None

        epochs = metrics.get("epochs")
        model = ModelVersion(
            id=model_id, name=(name or metrics.get("name") or path.stem), path=str(path.resolve()),
            created=metrics.get("created") or now_iso(), dataset=metrics.get("dataset"),
            base_weights=metrics.get("base_weights"), epochs=int(epochs) if isinstance(epochs, int) else None,
            map50=number_or_none("map50"), map50_95=number_or_none("map50_95"),
            per_class={key: value for key, value in (metrics.get("per_class") or {}).items() if isinstance(value, dict)},
        )
        data["models"].append(model.__dict__)
        if not data.get("active"):
            data["active"] = model.id
        _write_json(self.file, data)
        return model

    def set_active(self, model_id: str) -> None:
        data = self._load()
        if not any(entry.get("id") == model_id for entry in data["models"]):
            raise KeyError(model_id)
        data["active"] = model_id
        _write_json(self.file, data)

    def recommended(self) -> tuple[ModelVersion | None, str]:
        """The measured model with the best robot recall, then mAP50; with the reason in words."""
        measured = [model for model in self.versions() if model.measured and model.status != "Archived"]
        if not measured:
            return None, "No model has measured validation metrics yet."

        def key(model):
            recall = model.class_metric("robot", "recall")
            return (recall if recall is not None else -1.0, model.map50 or 0.0)

        best = max(measured, key=key)
        recall = best.class_metric("robot", "recall")
        if recall is not None:
            return best, f"Highest robot recall ({recall:.2f}), then mAP50 ({best.map50:.2f}), on its validation set."
        return best, f"Highest mAP50 ({best.map50:.2f}); robot recall was not reported."


def remove_file_to_trash(path: Path) -> bool:
    """Move a file to the recycle bin; returns False when the platform cannot."""
    from PySide6.QtCore import QFile
    result = QFile.moveToTrash(str(path))
    # PySide returns (ok, path_in_trash) on some versions and a bool on others.
    return bool(result[0] if isinstance(result, tuple) else result)
