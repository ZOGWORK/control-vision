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
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control_vision.pipeline.live_score import propose_events, read_live_overlay, summarize  # noqa: E402

FIELDS = ("auto_tips", "teleop_tips", "foul_credit")


def _official(official_dir: Path, event_code: str, number: int) -> dict | None:
    path = official_dir / f"{event_code}.json"
    if not path.is_file():
        return None
    matches = json.loads(path.read_text(encoding="utf-8"))
    match = next((m for m in matches if m.get("tournamentLevel") == "Quals" and m.get("id") == number), None)
    if not match or not match.get("scores"):
        return None
    result = {}
    for alliance in ("red", "blue"):
        s = match["scores"][alliance]
        result[alliance] = {"auto_tips": s["autoHiveTips"], "teleop_tips": s["dcHiveTips"],
                            "foul_credit": s["penaltyPointsByOpp"], "total": s["totalPoints"]}
    return result


def read(footage: Path, official_dir: Path, out_dir: Path, only: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(footage / "FOOTAGE_MANIFEST.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if only and not any(part in row["file"] for part in only):
            continue
        target = out_dir / (Path(row["file"]).stem + ".json")
        if target.exists():
            continue
        match = re.search(r"Qualification (\d+)", row["match_label_on_screen"])
        if not match:
            continue
        result: dict = {"file": row["file"], "event_code": row["event_code"], "match": int(match.group(1))}
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
        target.write_text(json.dumps(result), encoding="utf-8")
        print(row["file"], result.get("error", "ok"), flush=True)


def report(results_dir: Path) -> None:
    lines = ["| Match | Alliance | Live: AUTO TIPs / TELEOP TIPs / foul credit | Official | Last live score | Official total |",
             "|---|---|---|---|---|---|"]
    failed = []
    exact = {field: 0 for field in FIELDS}
    compared = 0
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "error" in result:
            failed.append(f"{path.stem}: {result['error']}")
            continue
        if result.get("complete") is False:
            failed.append(f"{path.stem}: read stopped early ({result.get('stopped_early')}); compared anyway")
        official = result.get("official")
        if not official:
            failed.append(f"{path.stem}: no official result")
            continue
        summary = summarize(propose_events(result["samples"]))
        for alliance in ("red", "blue"):
            got = tuple(summary[alliance][field] for field in FIELDS)
            want = tuple(official[alliance][field] for field in FIELDS)
            compared += 1
            for field, a, b in zip(FIELDS, got, want):
                exact[field] += a == b
            mark = "" if got == want else " ✗"
            lines.append(f"| {result['event_code']} Q{result['match']} | {alliance} | {' / '.join(map(str, got))}{mark} | "
                         f"{' / '.join(map(str, want))} | {summary[alliance]['last_stable_score']} | {official[alliance]['total']} |")
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
    if len(sys.argv) >= 5 and sys.argv[1] == "read":
        read(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:])
    elif len(sys.argv) == 3 and sys.argv[1] == "report":
        report(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
