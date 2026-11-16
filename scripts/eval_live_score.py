"""Measure live-overlay event proposals against official match results.

Two steps, so the slow OCR pass runs once and the comparison can be redone
whenever the event logic changes:

  python scripts/eval_live_score.py read <footage_dir> <official_dir> <results_dir> [clip-substring ...]
  python scripts/eval_live_score.py report <results_dir> > report.md

<footage_dir> holds FOOTAGE_MANIFEST.csv (columns file, event_code,
match_label_on_screen) and the clips. <official_dir> holds one JSON file per
event code with that event's matches, as returned by FTCScout's public REST
API (/rest/v1/events/<season>/<code>/matches). Fetching them is left to the
user. Only qualification matches are compared.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control_vision.pipeline.live_score import propose_events, read_live_overlay, summarize  # noqa: E402

FIELDS = ("auto_tips", "teleop_tips", "foul_credit")


def _official(official_dir: Path, event_code: str, number: int) -> dict | None:
    """The official result of one qualification match, or None when it is missing or the file is malformed (a bad
    official file must not stop the read of every other clip)."""
    path = official_dir / f"{event_code}.json"
    if not path.is_file():
        return None
    try:
        matches = json.loads(path.read_text(encoding="utf-8"))
        match = next((m for m in matches if m.get("tournamentLevel") == "Quals" and m.get("id") == number), None)
        if not match or not match.get("scores"):
            return None
        result = {}
        for alliance in ("red", "blue"):
            s = match["scores"][alliance]
            result[alliance] = {"auto_tips": s["autoHiveTips"], "teleop_tips": s["dcHiveTips"],
                                "foul_credit": s["penaltyPointsByOpp"], "total": s["totalPoints"]}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:  # JSONDecodeError is a ValueError
        print(f"Official results {path.name} are unreadable for Q{number}: {exc!r}", file=sys.stderr, flush=True)
        return None
    return result


def _write_json(path: Path, data: dict) -> None:
    # Write a temporary file and rename it, so an interrupted run never leaves a truncated result behind.
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temp, path)


def _load_json(path: Path) -> dict | None:
    """A cached result, or None when the file is unreadable: not valid JSON (for example truncated by an older
    version), or missing the fields every written result has."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # unreadable path (e.g. a directory), JSONDecodeError, UnicodeDecodeError
        return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("error"), str):
        return data  # a failed read; older versions wrote only file and error
    if isinstance(data.get("event_code"), str) and isinstance(data.get("match"), int) \
            and isinstance(data.get("samples"), list) and all(map(_valid_sample, data["samples"])):
        return data
    return None


def _number(value, kind=(int, float)) -> bool:
    # json.loads accepts NaN and Infinity; a reading never has them.
    return isinstance(value, kind) and not isinstance(value, bool) and math.isfinite(value)


def _valid_sample(sample) -> bool:
    """One overlay reading as read_live_overlay writes it: a time, and each counter a number or None."""
    return isinstance(sample, dict) and _number(sample.get("t_s")) \
        and all(sample.get(key) is None or _number(sample[key], int) for key in _SAMPLE_COUNTERS) \
        and (sample.get("clock_s") is None or _number(sample["clock_s"]))


_SAMPLE_COUNTERS = ("red_score", "blue_score", "red_tips", "blue_tips")


def _result_names(rows: list[dict]) -> dict[str, str]:
    """A distinct result file name per manifest file. The name stem alone, as before, when no other file gets the
    same name; files that would share one get the stem plus a short hash of their path, repeated until every name
    is unique (case-insensitively, as on Windows)."""
    files = list(dict.fromkeys(row["file"] for row in rows))
    names = {file: Path(file).stem + ".json" for file in files}
    salt = 0
    while True:
        taken: dict[str, list[str]] = {}
        for file, name in names.items():
            taken.setdefault(name.lower(), []).append(file)
        clashes = [group for group in taken.values() if len(group) > 1]
        if not clashes:
            return names
        for group in clashes:
            for file in group:
                digest = hashlib.sha1(f"{file}#{salt}".encode("utf-8")).hexdigest()[:8]
                names[file] = f"{Path(file).stem}-{digest}.json"
        salt += 1


def _clip_identity(path: Path) -> dict | None:
    """Size and modification time of the clip, so a file replaced under the same name is read again."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read(footage: Path, official_dir: Path, out_dir: Path, only: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(footage / "FOOTAGE_MANIFEST.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = _result_names(rows)
    for row in rows:
        if only and not any(part in row["file"] for part in only):
            continue
        match = re.search(r"Qualification (\d+)", row["match_label_on_screen"])
        if not match:
            continue
        result: dict = {"file": row["file"], "event_code": row["event_code"], "match": int(match.group(1))}
        target = out_dir / names[row["file"]]
        cached = _load_json(target) if target.exists() else None
        clip = _clip_identity(footage / row["file"])
        result["clip"] = clip
        same_clip = cached is not None and all(cached.get(key) == result[key] for key in ("file", "event_code", "match"))
        # A result written before clip identities were recorded has none; it is kept and given this one.
        same_footage = cached is not None and (cached.get("clip") is None or clip is None or cached["clip"] == clip)
        if same_clip and same_footage and "error" not in cached and cached.get("complete") is not False:
            # Keep the read, but pick up an official result added since.
            official = _official(official_dir, result["event_code"], result["match"])
            if official != cached.get("official") or (clip is not None and cached.get("clip") is None):
                cached["official"] = official
                if clip is not None:
                    cached["clip"] = clip
                _write_json(target, cached)
            continue
        # A failed, stopped-early or unreadable result is read again, and so is one written for a different
        # manifest row (a corrected event or match label) or for a clip file that has since been replaced.
        started = time.time()
        try:
            reading = read_live_overlay(footage / row["file"])
        except (RuntimeError, ValueError) as exc:
            result["error"] = str(exc)
        else:
            result.update(seconds=round(time.time() - started, 1), layout=reading["layout"],
                          samples=reading["samples"], complete=reading["complete"],
                          stopped_early=reading["stopped_early"])
        result["official"] = _official(official_dir, row["event_code"], result["match"])
        _write_json(target, result)
        print(row["file"], result.get("error", "ok"), flush=True)


def report(results_dir: Path) -> None:
    lines = ["| Match | Alliance | Live: AUTO TIPs / TELEOP TIPs / foul credit | Official | Last live score | Official total |",
             "|---|---|---|---|---|---|"]
    failed = []
    exact = {field: 0 for field in FIELDS}
    compared = 0
    for path in sorted(results_dir.glob("*.json")):
        result = _load_json(path)
        if result is None:
            failed.append(f"{path.stem}: result file is unreadable; run `read` again to redo it")
            continue
        if "error" in result:
            failed.append(f"{path.stem}: {result['error']}")
            continue
        if result.get("complete") is False:
            # Events after the cut are missing; comparing would understate them.
            failed.append(f"{path.stem}: read stopped early ({result.get('stopped_early')})")
            continue
        official = result.get("official")
        if not official:
            failed.append(f"{path.stem}: no official result")
            continue
        try:
            # Compute both alliances before counting either, so a malformed file adds nothing.
            summary = summarize(propose_events(result["samples"]))
            rows = []
            for alliance in ("red", "blue"):
                got = tuple(summary[alliance][field] for field in FIELDS)
                want = tuple(official[alliance][field] for field in FIELDS)
                rows.append((alliance, got, want, summary[alliance]["last_stable_score"], official[alliance]["total"]))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            failed.append(f"{path.stem}: result file is malformed ({exc!r}); run `read` again to redo it")
            continue
        for alliance, got, want, last_score, total in rows:
            compared += 1
            for field, a, b in zip(FIELDS, got, want):
                exact[field] += a == b
            mark = "" if got == want else " ✗"
            lines.append(f"| {result['event_code']} Q{result['match']} | {alliance} | {' / '.join(map(str, got))}{mark} | "
                         f"{' / '.join(map(str, want))} | {last_score} | {total} |")
    print(f"Alliance results compared: {compared}\n")
    for field in FIELDS:
        print(f"- {field.replace('_', ' ')}: {exact[field]} / {compared} exact")
    if failed:
        print("\nNot compared:\n")
        for item in failed:
            print(f"- {item}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    # The report marks mismatches with ✗; a Windows console's default code page cannot print it.
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) >= 5 and sys.argv[1] == "read":
        read(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:])
    elif len(sys.argv) == 3 and sys.argv[1] == "report":
        report(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
