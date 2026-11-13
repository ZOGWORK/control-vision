# Control Vision

Local-first FTC match video analyzer: track robots, count cycles, clip highlights, and score matches from video.

> Status: in development. Holdout performance has not been measured.

## What it is

Control Vision is part of the Control family (Control Point, Control Scout, Control Kit).
You upload a video of an FTC match and it tells you what happened — per robot, not just per alliance.

## The two main features

- **Practice analyzer** — track seeded robots, show top-down trails, speed and time in user-drawn zones, derive zone-trip cycle candidates, and create reviewable highlight clips.
- **Match scoring workflow** — review candidate events, assign event types and points, confirm robot attribution, and compare confirmed video points with official alliance scores. Automatic achievement inference is not implemented yet.

## How it works

- Detection: YOLO11n.
- Tracking: BoT-SORT motion tracking. OSNet ReID and tracklet stitching are not wired yet.
- Field mapping: one homography per video (click the 4 field corners once) to a top-down 12×12 ft field.
- Cycles: user-drawn pickup-to-scoring zone trips, medium confidence until ball detection is available.
- Scoring: zone trips create flagged, zero-point candidates. Optional scoring-element detections can suggest count changes inside a user-drawn ROI. A human must review the video and enter points.
- Runs fully offline on your own machine. No API calls. No cloud GPU.

## Docs

- [Build brief for Claude](docs/BUILD_BRIEF.md) — the full spec: stack, papers, prior art, training data, holdout set, build phases.
- [BIOBUZZ footage list](docs/BIOBUZZ_FOOTAGE.md) — 26 confirmed YouTube URLs (~15 events, ~300+ matches) for training and verification.
- [User guide](docs/USER_GUIDE.md) — project setup, calibration, zones, review, and exports.

## Run locally

Requires Python 3.11+ and a local YOLO11n ONNX model trained for the `robot` class. `ffprobe` is used when available; OpenCV provides a metadata fallback.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m control_vision
```

Create a project folder, import a video, click its four field corners, seed each robot identity, then select the local model and process. Videos stay at their original paths; project records and run logs are stored in the project folder. No pretrained model is bundled, and the app does not download weights.

The current build includes the tracking foundation, user-drawn zones, zone-trip cycles, reviewable event candidates, FFmpeg highlights, statistics, Scout JSON and robot-photo exports, and local training utilities. See the user guide for the manual review boundaries. Holdout acceptance metrics are still unmeasured; no accuracy or speed claim is implied.

## Workflow

- Every change via PR. Code review must give 5/5 before merge. Never merge below 5/5.
- One feature per PR. Base main only. No stacked PRs.
