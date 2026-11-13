# Control Vision

Local-first FTC match video analyzer: track robots, count cycles, clip highlights, and score matches from video.

> Status: Phase 1 foundation in development. Holdout performance has not been measured.

## What it is

Control Vision is part of the Control family (Control Point, Control Scout, Control Kit).
You upload a video of an FTC match and it tells you what happened — per robot, not just per alliance.

## The two main features

- **Practice analyzer** — detect and track every robot for the full match, show top-down trails, speed, and time in each zone, count cycles per robot, and auto-clip highlights around scoring moments.
- **Match scoring from video** — score a full match from video alone and attribute points to the correct robot, cross-checked against official scores.

## How it works

- Detection: YOLO11n.
- Tracking: BoT-SORT with OSNet ReID, plus offline tracklet stitching.
- Field mapping: one homography per video (click the 4 field corners once) to a top-down 12×12 ft field.
- Cycles: zone-crossing on top-down tracks, confirmed with a goal-ROI ball-count delta.
- Runs fully offline on your own machine. No API calls. No cloud GPU.

## Docs

- [Build brief for Claude](docs/BUILD_BRIEF.md) — the full spec: stack, papers, prior art, training data, holdout set, build phases.
- [BIOBUZZ footage list](docs/BIOBUZZ_FOOTAGE.md) — 26 confirmed YouTube URLs (~15 events, ~300+ matches) for training and verification.

## Run locally

Requires Python 3.11+ and a local YOLO11n ONNX model trained for the `robot` class. `ffprobe` is used when available; OpenCV provides a metadata fallback.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m control_vision
```

Create a project folder, import a video, click its four field corners, seed each robot identity, then select the local model and process. Videos stay at their original paths; project records and run logs are stored in the project folder. No pretrained model is bundled, and the app does not download weights.

Phase 1 currently implements video probing, manual corner calibration, identity seeding, offline robot detection/tracking, project-local tracks, and a top-down trail view. OSNet ReID, calibration reuse, editing/training screens, and acceptance measurements remain open work. No later phase has been started.

## Workflow

- Every change via PR. Code review must give 5/5 before merge. Never merge below 5/5.
- One feature per PR. Base main only. No stacked PRs.
