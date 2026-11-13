# Control Vision

Local-first FTC match video analyzer: track robots, count cycles, clip highlights, and score matches from video.

> Status: in development. README, description, and structure will be edited as the build takes shape.

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

## Workflow

- Every change via PR. Code review must give 5/5 before merge. Never merge below 5/5.
- One feature per PR. Base main only. No stacked PRs.

## License

Control Vision is licensed under the [GNU Affero General Public License v3.0](LICENSE). The complete corresponding source for a distributed build is available in this repository; release artifacts must identify the source commit used to build them.

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is offered under AGPL-3.0 or an Enterprise license. Any released model weights must include their own provenance and licensing information.
