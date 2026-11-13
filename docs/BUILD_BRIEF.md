Control Vision — Build Brief
Agent-general. ChatGPT or Claude Code — either agent can build from this brief.
Sushil picks which agent works on each task, and either agent can take over from the other.
See "Working with two agents" below.
How to use this brief
This brief is the full spec. Build from it end to end.
If anything here conflicts with the research page, this brief wins.
If a decision in this brief is wrong, say so in the PR description and propose the fix.
Do not shrink scope silently. If a requirement cannot be built, report it as a blocker.
Bruno is the PM. Decisions in the "Decisions made" section are final unless Sushil overrides them.
Sushil owns product direction. The coding agent builds the code.
Sushil pastes briefs himself. The agent never talks to Bruno directly — all direction flows through Sushil and this brief.
If you are taking over from another agent, read this brief, the repo state (main + open PRs), and the latest run notes before writing code.
Do not redo finished phases. Do not contradict a decision in this brief without flagging it to Sushil.
Working with two agents
Two agents may work on this project: ChatGPT and Claude Code.
Sushil decides which agent takes each task. Either agent can take over mid-build.
The agent with repo access works in PRs (see "Repo and workflow").
The agent without repo access delivers complete changed files with exact paths, plus a PR description Sushil can paste.
Never claim something is merged or live unless you verified it.
Keep docs/BUILD_BRIEF.md in sync with this brief — update it in your first PR if it differs.
Decisions made (Bruno, Nov 10, 2026)
App: PySide6 (Qt) desktop app, single Python codebase.
Rationale: one stack, native file dialogs, video frames shown in Qt widgets, no web server needed.
Tauri + Python sidecar is deferred to the product release. Do not build it now.
Language: Python 3.11+.
Storage: SQLite, one .db per project. No server. No cloud.
Export: JSON (Scout), CSV (stats), MP4 clips (FFmpeg), PNG stills (robot photos).
Detector: YOLO11n for v1. YOLO11s only if accuracy needs it.
Tracker: BoT-SORT with OSNet ReID. OC-SORT is the fallback.
Processing: offline, 5–10 fps sampling, with progress bar and cancel.
Identity: manual seeding at match start, first-class UI.
Zones: user-drawn polygons on the top-down map, stored in config.
Defaults come from the BIOBUZZ game manual. Never invent zone coordinates.
Scoring: cycles are reliable, report them plainly.
Points are reported with confidence and a human review queue.
Weights: versioned in models/ with version, date, and dataset hash.
Repo: ZOGWORK/control-vision, private. Created Nov 10, 2026, made private Nov 11, 2026.
README, this brief, and the footage list are already on main.
Purpose
Build Control Vision end to end.
Control Vision is a local-first downloadable app.
It analyzes FTC match video: tracks robots, counts cycles, clips highlights, scores matches from video.
It is part of the Control family: Control Point, Control Scout, Control Kit, Control Vision.
Product overview — the user journey
User creates a project (one event, or one practice session).
User imports a match video (mp4, mov, mkv).
User calibrates once per video: clicks the 4 field corners, draws zones.
User seeds identity: clicks each robot in the first frame, assigns team number and alliance.
User presses Process. The app samples frames, detects, tracks, maps, counts.
User watches results: top-down map with trails, per-robot stats, cycle list, scoring timeline.
User reviews flagged moments in the review queue and confirms or corrects them.
User exports: highlights, stats (CSV/JSON), robot photos, Scout export.
Feature 1 — Practice analyzer (detailed requirements)
Import any video file (mp4, mov, mkv). Probe duration, fps, resolution with ffprobe.
Warn (do not block) if resolution is below 720p or fps below 15.
Calibration screen: user clicks the 4 field corners in order on a still frame.
Corners map to a top-down 12×12 ft field.
Save calibration per video. Reuse it for other videos from the same camera setup.
Zone editor: user draws polygons on the top-down map.
Zones: scoring zone (hive), pickup zones, park zones.
Zone set is stored in a config file, not hardcoded.
Defaults come from the BIOBUZZ field diagram in the game manual.
Identity seeding: on the first clear frame, user clicks each robot and assigns team number + alliance (red/blue).
1–2 robots in practice, 4 in a match. Support up to 4.
Processing: sample frames at 8 fps by default (configurable 5–10).
Show stage, percent, elapsed time, and a Cancel button.
Write a run log to the project folder.
Results dashboard:
Top-down field map with animated robot trails and a playable timeline.
Per-robot stat cards: cycles, average cycle time, top speed, distance traveled, time per zone.
Cycle list: each cycle with start/end time; click a cycle to jump the video to it.
Scoring timeline: every scoring event with time, robot, points, confidence.
Highlights: auto-clip around every scoring event (3 s before, 3 s after by default).
List clips with thumbnails. Preview in-app. Export selected clips as MP4.
Works with 1 robot (solo practice) as well as 2 and 4.
Feature 2 — Match scoring from video (detailed requirements)
Score a full 2:30 match from video alone.
Attribute points to the correct robot.
Show per-robot points, per-alliance totals, and the match total.
Cross-check against official scores from ftc-events.firstinspires.org.
User pastes the official score; the app shows video total vs official total and the difference.
Every scored event carries a confidence (high / medium / low) and a status (confirmed / flagged / rejected).
Flagged events go to the review queue with the clip attached.
The review queue is the source of truth for the final numbers.
A match is not "scored" until the user has cleared the review queue or explicitly accepted the flagged items.
Build this after Feature 1 works and is verified. Do not build both at once.
Target users
FTC teams at practice (1–2 robots, fixed camera).
FTC teams and scouts at competitions (4 robots, full match).
Control Scout users (video-derived stats feed Scout).
Screens
Home / Projects: project list, New Project, Open, recent videos.
Import: pick video file, probe results, enter event code and match number.
Calibrate: corner clicking, zone drawing, save/reuse calibration.
Identity: first-frame robot picking, team number + alliance assignment.
Processing: progress bar, current stage, log lines, Cancel.
Results: top-down map + trails + timeline, per-robot stat cards, cycle list, scoring timeline.
Highlights: clip list with thumbnails, preview player, export.
Review queue: flagged moments with clip, Confirm / Reject / Reassign robot.
Export: JSON (Scout), CSV (stats), clips, photos.
Settings: model path, confidence threshold, sample fps, FFmpeg path.
Every screen must work with mouse and keyboard. No hidden controls.
Data model (SQLite, one .db per project)
projects (id, name, created_at).
videos (id, project_id, path, duration_s, fps, width, height, event_code, match_number).
calibrations (video_id, corners_json, zones_json, created_at).
corners_json: 4 (x, y) pixel points in order.
zones_json: named polygons in top-down feet.
robots (id, video_id, team_number, alliance, color_hint).
alliance is red or blue. color_hint is the bumper color sample.
tracks (id, robot_id, frame, t_s, x_ft, y_ft, conf).
One row per robot per sampled frame.
x_ft, y_ft are top-down field feet, origin at one field corner.
cycles (id, robot_id, start_t_s, end_t_s, pickup_zone, scoring_zone).
scoring_events (id, robot_id, t_s, points, confidence, status, source).
confidence: high, medium, low.
status: confirmed, flagged, rejected.
source: zone_crossing, goal_roi, scoreboard_ocr.
clips (id, scoring_event_id, path, start_t_s, end_t_s).
photos (id, robot_id, path, t_s).
Never store video files in the database. Store paths only.
Export the whole match as one JSON file (schema in the Scout section).
Tech stack — decided
Python 3.11+.
GUI: PySide6 (Qt).
Detection: Ultralytics YOLO11 (YOLO11n weights for v1).
Tracking: BoT-SORT implementation with OSNet ReID.
Fallback tracker: OC-SORT.
Offline stitching: StrongSORT AFLink + GSI.
Video IO: OpenCV. Probing: ffprobe. Clipping: FFmpeg.
Inference: ONNX Runtime on CPU by default.
OpenVINO on Intel, optional TensorRT on NVIDIA.
Storage: SQLite (stdlib sqlite3).
Packaging: PyInstaller for v1.
Entry point: python -m control_vision.
requirements.txt pins every dependency with versions.
Dev setup: python -m venv .venv, pip install -r requirements.txt, python -m control_vision.
No web server. No cloud calls. No API keys.
Project structure
control_vision/
main.py (entry point).
app/ (PySide6 windows, one file per screen).
pipeline/ (ingest, detect, track, homography, zones, cycles, scoring, clips).
training/ (dataset tools, training scripts, ONNX export).
scout/ (Scout JSON export, photo export).
storage/ (SQLite schema, queries).
config/ (zone defaults, thresholds).
models/ (weights, versioned, with a README noting version + date + dataset hash).
tests/ (unit tests for zone logic, cycle state machine, homography).
docs/
BUILD_BRIEF.md (this file).
BIOBUZZ_FOOTAGE.md (training footage list).
data/ (gitignored: videos, datasets, run outputs).
README.md, requirements.txt, .gitignore.
Pipeline — step by step
Probe: ffprobe reads duration, fps, resolution.
Sample: extract frames at 8 fps (configurable 5–10).
Detect: YOLO11n ONNX, imgsz 640, conf 0.25, NMS IoU 0.5.
Default thresholds are starting points. Tune on validation data, then update this brief.
Track: BoT-SORT, track buffer 30 frames, match threshold 0.8, OSNet ReID.
Stitch offline: AFLink + GSI joins tracklets broken by occlusion.
Homography: 4 clicked corners → 12×12 ft top-down.
Project the bottom-centre of each robot box.
Bottom-centre is the ground contact point. Do not use box centre.
Smooth: moving average (window 5 frames) on top-down positions to kill jitter.
Compute speed from smoothed positions.
Zones: point-in-polygon per frame, debounce 0.5 s to avoid flicker on edges.
Cycle state machine, per robot:
EMPTY → (enters pickup zone) → AT_PICKUP → (ball detected near robot) → CARRYING → (enters scoring zone) → SCORED → EMPTY.
Count one cycle on SCORED.
If ball detection is not available yet, count zone trips (pickup → scoring) and mark confidence medium.
Scoring: goal-ROI ball-count delta + scoreboard OCR + reconciliation (see Scoring section).
Highlights: for each scoring event, clip event time −3 s to +3 s with FFmpeg.
Prefer stream copy when the container allows it; re-encode only when needed.
Photos: per robot, pick the best frame (largest box + highest confidence).
Crop with 20% margin. Export PNG.
Detection classes
robot.
pollen ball.
nectar ball.
Class names must be verified against the BIOBUZZ game manual before labeling.
Do not add classes without a manual reference.
Do not hardcode colors. Color is a feature, not a class.
Zones
Zones are user-drawn polygons on the top-down map.
Stored in calibrations.zones_json, per video.
Zone names come from the BIOBUZZ field: scoring zone (hive), pickup zones, park zones.
Zone geometry comes from the official BIOBUZZ field diagram.
Never invent zone coordinates. If the manual diagram is unclear, ask Sushil.
Zone config file (config/zones_biobuzz.json) holds the defaults.
One file per season. A new season gets a new file, never an edit that breaks the old one.
Training pipeline
Dataset layout: datasets/biobuzz-v1/images + labels, split train/val.
data.yaml names the classes and paths.
Labeling: CVAT with video track interpolation (label once, interpolate across frames).
Alternatives: Roboflow Annotate (SAM Smart Polygon), Label Studio.
Augmentation: mosaic, flip, scale.
Keep HSV/color augmentation light. Color matters for ball classes.
Hyperparameters (starting point, Ultralytics defaults — tune on val, then update this brief):
imgsz 640, epochs 100, batch 16, lr 0.01, patience 20.
Metrics: mAP50 and mAP50-95 per class, on the val split.
Targets (not measured results): mAP50 ≥ 0.85 for robot, ≥ 0.75 for balls.
Export: ONNX (opset 12 or newer). Test the ONNX file in ONNX Runtime before shipping it.
Weights versioning: models/yolo11n-biobuzz-v1.0.0.onnx.
Each version records date, dataset hash, training config, and val metrics in models/README.md.
Old weights are kept, never overwritten.
Retraining for a new season: new dataset + new weights version.
Free Colab T4 is enough for YOLO11n/s fine-tuning.
12 h max session, ~90 min idle limit, unpublished quota.
Kaggle ~30 GPU-hours/week as backup.
Verification protocol
Verify on the holdout set only. Never on training data.
Holdout: STL North Nov 14, Daisy Meet 2 Nov 21, Michigan Midland Nov 20–21 (or MI States Dec 11–13).
Ground truth: official scores on ftc-events.firstinspires.org.
Hand-count cycles for at least 5 holdout matches as the cycle ground truth.
Report every metric with the match list it was measured on.
Never claim an accuracy number without the measurement behind it.
Acceptance targets (targets set by Bruno — not measured results)
Phase 1 (tracking):
≤ 2 ID switches per match on average, on the holdout set.
Track completeness ≥ 95% of frames where the robot is visible.
Homography reprojection error < 5 px on the calibration frame.
Phase 2 (cycles):
Cycle count within ±1 of hand-counted ground truth on ≥ 90% of holdout matches.
Phase 3 (highlights):
Every scoring event has a clip. 100%.
Clips start 3 s before the event and end 3 s after.
Phase 4 (scoring):
Match total within ±10% of the official score on ≥ 80% of holdout matches.
Per-robot attribution correct on ≥ 80% of scored events (verified by human review of clips).
If a phase misses its targets, fix the phase. Do not move on and do not lower the targets silently.
Control Scout link
Export one JSON file per match. Schema (v1):
app: "control-vision".
version: "1.0.0".
event_code, match (e.g. "Q12"), video filename.
teams: array of team_number, alliance, cycles, cycle_time_avg_s, zone_time_s, top_speed_ft_s, distance_ft, points_video, points_official, confidence.
clips: array of event time, robot, points, clip filename.
photos: array of team_number, photo filename.
Robot photos: crop from the best frame (largest box + highest confidence), 20% margin, PNG.
Minimum useful size: 200×200 px. Skip smaller crops and say so in the log.
Scout consumes the JSON by manual upload in v1. API integration comes later.
Same versioned numbers everywhere.
Computation never forks, presentation stays free (Scout rule).
Performance budgets
A 2:30 match at 8 fps (1,200 frames) processes in ≤ 10 minutes on a mid laptop CPU (i5 / Ryzen 5, no GPU).
App startup ≤ 5 s.
Video import (probe + first-frame thumbnail) ≤ 10 s.
Results dashboard opens in ≤ 3 s for a processed match.
Measure on the target laptop and report the numbers. Budgets are targets, not guesses.
Error handling and logging
Every run writes a log file to the project folder (run timestamp in the name).
Log: video probed, frames sampled, detections per frame, tracks created, ID switches, cycles, scoring events, errors.
If detection finds 0 robots in the first 30 sampled frames, stop and tell the user (wrong video, wrong model, or bad calibration).
If the video file is corrupt or unreadable, fail with a plain message naming the file.
If FFmpeg is missing, tell the user where to get it. Do not fail silently.
Never crash on one bad frame. Skip it, log it, continue.
Never show a stack trace to the user. Log it, show one plain sentence.
Papers to read
Paper
What it gives Control Vision
Link
YOLO11 review
Detector baseline, CPU numbers
https://arxiv.org/abs/2510.09653
YOLOv12
Newer detector, A/B test
https://arxiv.org/abs/2502.12524
RF-DETR nano
Detector A/B test
https://arxiv.org/abs/2511.09554
ByteTrack
Tracking baseline, low-confidence recovery
https://arxiv.org/abs/2110.06864
BoT-SORT
Best CPU-friendly tracker
https://arxiv.org/abs/2206.14651
OC-SORT
Pure-motion fallback
https://arxiv.org/abs/2203.14360
StrongSORT
Offline tracklet stitching
https://arxiv.org/abs/2202.13514
Deep-EIoU
Sports-proven upgrade
https://arxiv.org/abs/2306.13074
SAM 2
Segmentation tracking, labeller
https://arxiv.org/abs/2408.00714
SAM 3
Prompted segmentation, labeller
https://arxiv.org/abs/2511.16719
SportsMOT
Sports tracking benchmark
https://arxiv.org/abs/2304.05170
SoccerNet-GSR
Game-state reconstruction
https://arxiv.org/abs/2404.11335
PlayerTV
OCR identity voting
https://arxiv.org/abs/2407.16076
Key numbers (verify before repeating them)
YOLO11n: 39.5% COCO mAP, 56.1 ms CPU-ONNX.
YOLOv8n: 37.3%, 80.4 ms.
ByteTrack: 80.3 MOTA on MOT17.
BoT-SORT: 1,257 ID switches vs ByteTrack 2,196 (same benchmark).
OC-SORT: 700+ fps CPU association.
Deep-EIoU: 77.2 HOTA on SportsMOT.
Offline 5–10 fps is acceptable. Do not promise real-time.
Prior art — what to copy, what to avoid
FRC Team 334 (Chief Delphi).
Link: https://www.chiefdelphi.com/t/building-ai-assisted-frc-scouting-what-worked-what-failed-and-what-we-learned/522442
Worked: ball tracking (SAM 3 prompted "yellow ball" best; HSV adequate only in clean venues).
Failed: robot tracking (object permanence, ID loss on occlusion, side-camera variability).
Shipped: hybrid — human traces the robot, system counts shots automatically.
Copy this hybrid. Do not repeat the fully-autonomous attempt first.
FRC Team 1294 Auto Scout.
Post: https://www.chiefdelphi.com/t/introducing-team-1294-s-auto-scout-program-the-p-a-c-k/516763
Code: https://github.com/packofparts/scouting-ai
Per-event corner calibration, Roboflow detector, Hungarian association, manual team numbers.
Explicitly does NOT detect points scored.
Users report path and start-position errors.
Copy: per-event calibration, manual team numbers.
Avoid: claiming scoring it does not do.
Nomadic Dragons FTC 25528 Auto Judge.
Site: https://www.nomadic-dragons.com/
Claims "first-ever computer vision auto judge in FTC", real-time element counting.
No code, no demo, no accuracy numbers found. UNVERIFIED.
Do not treat it as proven. Do not copy its claims.
Critical correction — game pieces
BIOBUZZ 2026-27 game pieces are ~3-inch pollen balls.
They are NOT purple/green artifacts.
Purple/green artifacts are DECODE 2025-26.
Pollen ball colors were not stated in the official preview.
Verify colors from the BIOBUZZ game manual before labeling anything.
Do not hardcode purple/green anywhere.
Do not hardcode any color until verified.
Training data
BIOBUZZ footage only. DECODE footage excluded.
26 confirmed YouTube URLs, ~15 events, ~300+ matches.
Full list in the appendix at the end of this brief.
Best starter: FUN Robot in 30 Hours clipped matches.
Scores are in the video descriptions = free ground truth.
Biggest chunk: Michigan FiM AV qualifiers (~200+ matches).
Ground truth: official scores on ftc-events.firstinspires.org.
No hand-labeling needed for verification.
Camera angle is UNVERIFIED for every video.
Watch a few minutes of each before training.
Confirm fixed full-field view.
Holdout test set — do NOT train on these
FTC STL North League Meet, Battle High School, Nov 14, 2026.
Stream: https://www.youtube.com/watch?v=EHLGE5Rl3qU
Daisy League Meet 2, Newton IA, Nov 21, 2026.
FTC Iowa channel. URL not published yet.
Michigan Midland Qualifier, Nov 20–21, 2026.
Or MI States, Dec 11–13, 2026.
Alternative holdout: FIRST Chesapeake Washington DC, Dec 5–6, 2026.
Identity — how to keep the right robot the right robot
First: alliance color.
Red/blue bumper histogram. Robust. Halves the problem.
Second: ReID + position to maintain identity across frames.
Third: team-number OCR only as confirmation. It is fragile.
Vote OCR over top-20 crops offline (PlayerTV pattern).
Team 334 found a vision-language model beat classical OCR.
Iowa State got perfect counts in only 43% of matches.
On uncertainty: flag, do not swap.
A wrong ID is worse than an unknown ID.
Manual identity seeding and correction UI is a first-class feature.
Scoring from video
Primary signal: zone-crossing on top-down tracks.
Pickup → scoring = cycle.
Confirmation: goal-ROI ball-count delta.
Cross-check 1: scoreboard OCR (PaddleOCR, voted over frames).
Cross-check 2: official ftc-events totals for the match.
Reconciliation rule: if the three signals disagree, trust zone-crossing first, goal-ROI second, scoreboard OCR third.
Mark the event flagged and put it in the review queue.
No FTC scoreboard-OCR evidence exists. Treat it as unproven until measured.
Cycles are reliable. Per-robot point totals are the hardest part.
Report cycles with confidence.
Report points with their verification status.
Labeling and training sizes
Prototype: 50–100 images.
Reliable: 200–300 instances per class.
V1 target: 500–1,000 labeled frames (frames are instance-rich).
Production target (Ultralytics guidance): 1,500 images / 10,000 instances per class.
Free Colab T4 is enough for YOLO11n/s fine-tuning.
12 h max session, ~90 min idle limit, unpublished quota.
Kaggle ~30 GPU-hours/week as backup.
Build order — four phases, in order, verify each before the next
Phase 1: robot tracking + top-down trails. Fixed camera, 1–2 robots first.
Exit criteria: Phase 1 acceptance targets (see Verification section).
Phase 2: cycle counting via zone entries.
Exit criteria: Phase 2 acceptance targets.
Phase 3: highlight clipping with FFmpeg around detected events.
Exit criteria: Phase 3 acceptance targets.
Phase 4: scoring / element detection. Hardest. Optional.
Exit criteria: Phase 4 acceptance targets.
Verify each phase on the holdout set before moving on.
Never claim an accuracy number without measuring it on the holdout.
If a phase misses its targets, fix the phase. Do not move on. Do not lower targets silently.
Repo and workflow
Repo: ZOGWORK/control-vision, private. Created Nov 10, 2026.
URL: https://github.com/ZOGWORK/control-vision
README, this brief, and the footage list are already on main.
Every change via PR once building starts.
Loopy must give 5/5 before merge. Never merge below 5/5.
Commits authored as Sushil Muthuvelkumar (sushilm20@users.noreply.github.com).
One feature per PR. Base main only. No stacked PRs.
Aim 3–5 PRs per session, hard max 10. Group related changes into one PR.
Run the full local checks before pushing, so each push needs at most one review round.
If you cannot open a PR from your environment, deliver the complete changed files with exact paths, plus a PR description Sushil can paste.
Do not push straight to main.
Do not claim anything is merged or live unless you verified it.
Keep docs/BUILD_BRIEF.md in sync with this brief — update it in your first PR if it differs.
Open questions (do not guess — verify or ask Sushil)
Verified against official BIOBUZZ Competition Manual TU09 (Nov 12, 2026): POLLEN are yellow and approximately 2.8 in.; NECTAR are red or blue and approximately 3.6 in. Point values and scoring criteria from Section 10.5 are versioned in config/scoring_biobuzz.json. Source: https://ftc-resources.firstinspires.org/ftc/archive/2027/game/cm-html/BIOBUZZ%20Competition%20Manual%20-%20TU09.htm
Exact BIOBUZZ field geometry still needs to be digitized from the official field drawings; users draw polygons until then.
FTC scoreboard OCR: does it work on real streams? Unproven.
CPU fps figures are host-dependent. Measure on the target laptop.
No published FTC end-to-end scoring accuracy exists.
Control Vision would be the first verified one.
Deep HM-SORT 80.1 HOTA claim is disputed. Do not use it.
Which laptop is the reference machine for performance budgets? Ask Sushil.
Does the app need Windows only, or Mac too? Sushil's primary computer is Windows. Ask before adding a second platform.
Appendix — BIOBUZZ footage list (26 URLs)
No videos were downloaded. Download with yt-dlp as needed.
Ground truth for every event: official scores on ftc-events.firstinspires.org.
Michigan qualifiers (FiM AV)
Royal Oak, Nov 6–7, 28 teams.
Practice Day: https://www.youtube.com/watch?v=Jipe7rWPZWA
Day 1: https://www.youtube.com/watch?v=FGiUuys_-BE
Belding, Nov 6–7, 27 teams, 34 quals.
Day 1: https://www.youtube.com/watch?v=RE0KNvjR07c
Coloma, Nov 6–7, 31 teams.
Day 1: https://www.youtube.com/watch?v=ghEe-5fH_Kw
Hartland, Nov 6–7, Howell, 30 teams, 38 quals.
Practice Day: https://www.youtube.com/watch?v=JkprQksmRIg
Day 1: https://www.youtube.com/watch?v=A5n0Xamahk8
Bullock Creek, Nov 6–7, Midland, 34 teams, ~43 quals.
Practice Day: https://www.youtube.com/watch?v=2Y1fVAD-Lls
Day 1: https://www.youtube.com/watch?v=jXgqS-2PGvY
KILTS highlights only (not a full stream): https://www.youtube.com/watch?v=R1f0B7fM8g0
Alabama
Athens Qualifier, Oct 24, 2026.
Full-day stream: https://www.youtube.com/watch?v=4tScR8RexiA
Arizona
Quail Qualifier, Nov 7, Glendale, 25 teams, 32 quals + 10 playoffs.
https://www.youtube.com/watch?v=tLK-tRpZSTA
High Country Qualifier, Nov 7, Flagstaff, 18 teams.
https://www.youtube.com/watch?v=ITK-Rc5LvT4
California
NorCal Screamage, Nov 1, San Jose.
https://www.youtube.com/watch?v=ffYioYR0TQE
East Bay League Meet #1, Nov 1.
https://www.youtube.com/watch?v=44mne4zzkPU
Iowa
Daisy League Meet 1, Nov 7, Martensdale.
https://www.youtube.com/watch?v=EPhcvM19DnQ
Ohio
CSU & CMSD Scrimmage, Cleveland State / Cleveland Metro.
https://www.youtube.com/watch?v=s5-ACnmt5vk
FUN Robot in 30 Hours — Wisconsin, Sep 13–14, 2026
Full streams:
Saturday: https://www.youtube.com/watch?v=b79NPndUWKY
Sunday: https://www.youtube.com/watch?v=mnId-C15OTc
Clipped matches (scores in descriptions):
Match 3: https://www.youtube.com/watch?v=CNaUmOvj2WQ (Red 21, Blue 41)
Match 5: https://www.youtube.com/watch?v=k-mSVM3Foos (Red 33, Blue 44)
Match 6: https://www.youtube.com/watch?v=xdIbYp4WX0U (Red 38, Blue 11)
Match 7: https://www.youtube.com/watch?v=zTEYuu9a0y4 (feed cut last 15s)
Match 9: https://www.youtube.com/watch?v=EUdBXBLy7No (Red 19, Blue 35)
Match 12: https://www.youtube.com/watch?v=AS1vn-KhBJs (Red 25, Blue 52)
Georgia Tech Ri30H Sunday stream, Sep 13:
https://www.youtube.com/watch?v=2xiqGhEc7fc
Single-robot demo — low training value
10653 Dragatrons 222-pt solo world record (not a match).
https://www.youtube.com/watch?v=kwYU2RU7V14
