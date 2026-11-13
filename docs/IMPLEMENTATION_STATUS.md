# End-to-end implementation status

This is the scope ledger for the build brief. It separates implemented product behavior from model assets and validation that are not present in the repository. BIOBUZZ scoring references use the official 2026–27 Competition Manual TU09, updated Nov 12, 2026.

## Implemented in the desktop app

- Home / Projects entry screen, create/open project actions, and recent videos for the active SQLite project.
- Video import, ffprobe/OpenCV metadata, low-resolution/frame-rate warning, event and match metadata, video playback and seeking.
- Per-video four-corner field calibration, editable top-down polygons, saved camera calibration reuse, and manual robot/team/alliance seeding.
- Configurable 5–10 fps processing, model/confidence settings, FFmpeg path, stage/progress/cancel, project run logs, and per-frame failure logging.
- YOLO detector integration, BoT-SORT tracking, homography from robot ground-contact bottom center, smoothed tracks, zone time and movement statistics.
- Zone-trip cycle candidates, scoring-element ROI count-change candidates, confidence/status review queue, manual event classification and recipient-alliance selection.
- Candidate highlight generation, in-app playback, export, best-frame robot PNGs, CSV stats and Scout JSON.
- BIOBUZZ TU09 reference and configured scoring values, including separate match points and Ranking Points, phase timing, fouls as opponent credits, and manual confirmation of end-state achievements.
- Official score comparison and manual DQ/event threshold controls. Confirmed events are the only video-derived points included in totals; flagged events keep the score visibly pending.
- Offline scoreboard OCR using user-drawn red/blue ROIs, RapidOCR local models and multi-frame consensus. OCR values are reviewable proposals and require explicit user acceptance before comparison; these are not scored events.
- Optional local ReID model selection for BoT-SORT; absence is logged and appearance matching remains disabled.
- Local training utilities for frame extraction, dataset hashing, YOLO fine-tuning, versioned ONNX export, and ONNX Runtime CPU smoke check.

## Remaining release blockers

- There are no BIOBUZZ-trained detector or OSNet/ReID weights or labeled dataset in the repository. Processing requires a local detector model with class 0 `robot`. A generic COCO model is not represented as a valid substitute.
- The detector UI is described as ONNX-first, but Ultralytics' BoT-SORT `model.track` backend path has not been verified as ONNX Runtime CPU on target machines. No target-laptop measurement is available.
- ReID accepts a compatible user-supplied model; no OSNet architecture checkpoint has been bundled or validated.
- OC-SORT fallback and StrongSORT AFLink + GSI offline tracklet stitching are not implemented. Current tracking is BoT-SORT only.
- OCR reconciliation with zone/element evidence is not implemented. OCR proposes video scoreboard totals only; all point-event attribution remains human-reviewed. On the sampled Salk Q3 clip it returned 156–93 from 2 readable frames of 12, while the [official FIRST Q3 result](https://ftc-events.firstinspires.org/2026/USWASAM1/qualifications) is 180–103. The UI confidence is capped; this proposal is not an official result or a general accuracy claim.
- The seeded optical-flow mode is an experimental movement preview, not a full-match identity detector. A real Salk Q3 smoke run retained 201 points for one seed across a 170-second clip and produced one valid robot photo. Do not use these tracks as scoring evidence.
- The public repository now declares AGPL-3.0 in `LICENSE`, consistent with the chosen open-source path for Ultralytics YOLO. Before distributing a Windows build, include the license, identify the exact corresponding source commit, and publish provenance/licensing information for any fine-tuned weights. No BIOBUZZ weights are bundled yet. See [Ultralytics licensing](https://docs.ultralytics.com/help/contributing/#open-sourcing-your-yolo-project-under-agpl-30).
- Automatic scoring for HIVE, FLOWER ownership/bottom NECTAR, GARDEN, robot parking/LEAVE and referee calls is not implemented. ROI changes are only review candidates; visual and referee evidence must be interpreted by a human.
- Qualification team-level disqualification is not implemented. The current alliance-DQ control is for Playoff-style alliance results; it must not be used to model a Qualification team DQ, which does not disqualify its partner under T601.
- Model training and acceptance targets cannot be completed without suitable labeled BIOBUZZ data, fixed-camera footage screening, and held-out matches with hand-counted cycles / official results. No accuracy, speed, or holdout acceptance claim is made.
- The app includes a BIOBUZZ TU09 rule reference and manual-derived point/RP logic, but does not replace event referees, Robot Inspectors, the official scoring system, or later manual/team updates.

## Verification needed before release

Use the brief's holdout matches only for verification, never for training. Measure ID switches, track completeness, calibration reprojection error, cycle error, clip coverage, score error, attribution accuracy, startup/import/results latency, and processing time on the target hardware. The Windows laptop used for the Salk Q3 optical-flow workflow completed the 170-second, 2 fps preview smoke in 84.76 seconds. This does not measure the required 8 fps YOLO processing budget, so performance thresholds remain targets rather than results.

## Rule sources

- [Official BIOBUZZ Competition Manual TU09](https://ftc-resources.firstinspires.org/ftc/archive/2027/game/cm-html/BIOBUZZ%20Competition%20Manual%20-%20TU09.htm)
- [Official BIOBUZZ game archive and Team Updates](https://ftc-resources.firstinspires.org/ftc/archive/2027/game)
- Full paraphrased in-app guide: [`BIOBUZZ_RULES.md`](BIOBUZZ_RULES.md)

