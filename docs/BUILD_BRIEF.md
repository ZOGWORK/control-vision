# Control Vision — Build Brief for Claude Code

## Purpose

- Build Control Vision end to end.
- Control Vision is a local-first downloadable app.
- It analyzes FTC match video: tracks robots, counts cycles, clips highlights, scores matches from video.
- It is part of the Control family: Control Point, Control Scout, Control Kit, Control Vision.
- Sushil owns product direction. Claude builds the code.
- Sushil pastes briefs himself. Claude never talks to Bruno directly.

## The two main features

- Feature 1 — Practice analyzer.
  - Upload a practice or match video.
  - Detect and track every robot for the full match.
  - Show top-down trails, speed, time in each zone.
  - Count cycles per robot.
  - Auto-clip highlights around scoring moments.
- Feature 2 — Match scoring from video.
  - Score a full match from video alone.
  - Attribute points to the correct robot.
  - Cross-check against official scores.
  - This is the harder feature. Build it after Feature 1 works.

## Target users

- FTC teams at practice (1–2 robots, fixed camera).
- FTC teams and scouts at competitions (4 robots, full match).
- Control Scout users (video-derived stats feed Scout).

## Recommended stack (from research)

- Detector: YOLO11n for v1 (YOLO11s if accuracy needs it).
- Tracker: BoT-SORT with OSNet ReID.
- Fallback tracker: OC-SORT (pure motion, best when appearance fails).
- Offline stitching: StrongSORT AFLink + GSI to join tracklets after the match.
- Upgrade path if ID switches persist: Deep-EIoU.
- Field mapping: one homography per video.
  - User clicks the 4 field corners once.
  - Map to a top-down 12×12 ft field.
  - Project the bottom-centre of each robot box.
- Cycle counting: zone-crossing on top-down tracks.
  - Pickup zone → scoring zone → back = one cycle.
  - Confirm with a goal-ROI ball-count delta.
- Labeller only: SAM 3 (too slow on CPU, swaps IDs on identical robots).
- Identity seeding: manual at match start, first-class UI, not a fallback.
- Runtime: ONNX Runtime on CPU by default.
  - OpenVINO on Intel, optional TensorRT on NVIDIA.
- Training: free Colab T4 GPU.
- Labeling: CVAT (video track interpolation) or Roboflow Annotate or Label Studio.
- Video clipping: FFmpeg around detected events.
- Packaging: PyInstaller for v1, Tauri + Python sidecar for the product release.
- Runs fully offline. No API calls. No cloud GPU.

## Papers to read

| Paper | What it gives Control Vision | Link |
|---|---|---|
| YOLO11 review | Detector baseline, CPU numbers | https://arxiv.org/abs/2510.09653 |
| YOLOv12 | Newer detector, A/B test | https://arxiv.org/abs/2502.12524 |
| RF-DETR nano | Detector A/B test | https://arxiv.org/abs/2511.09554 |
| ByteTrack | Tracking baseline, low-confidence recovery | https://arxiv.org/abs/2110.06864 |
| BoT-SORT | Best CPU-friendly tracker | https://arxiv.org/abs/2206.14651 |
| OC-SORT | Pure-motion fallback | https://arxiv.org/abs/2203.14360 |
| StrongSORT | Offline tracklet stitching | https://arxiv.org/abs/2202.13514 |
| Deep-EIoU | Sports-proven upgrade | https://arxiv.org/abs/2306.13074 |
| SAM 2 | Segmentation tracking, labeller | https://arxiv.org/abs/2408.00714 |
| SAM 3 | Prompted segmentation, labeller | https://arxiv.org/abs/2511.16719 |
| SportsMOT | Sports tracking benchmark | https://arxiv.org/abs/2304.05170 |
| SoccerNet-GSR | Game-state reconstruction | https://arxiv.org/abs/2404.11335 |
| PlayerTV | OCR identity voting | https://arxiv.org/abs/2407.16076 |

## Key numbers (verify before repeating them)

- YOLO11n: 39.5% COCO mAP, 56.1 ms CPU-ONNX.
- YOLOv8n: 37.3%, 80.4 ms.
- ByteTrack: 80.3 MOTA on MOT17.
- BoT-SORT: 1,257 ID switches vs ByteTrack 2,196 (same benchmark).
- OC-SORT: 700+ fps CPU association.
- Deep-EIoU: 77.2 HOTA on SportsMOT.
- Offline 5–10 fps is acceptable. Do not promise real-time.

## Prior art — what to copy, what to avoid

- FRC Team 334 (Chief Delphi).
  - Link: https://www.chiefdelphi.com/t/building-ai-assisted-frc-scouting-what-worked-what-failed-and-what-we-learned/522442
  - Worked: ball tracking (SAM 3 prompted "yellow ball" best; HSV adequate only in clean venues).
  - Failed: robot tracking (object permanence, ID loss on occlusion, side-camera variability).
  - Shipped: hybrid — human traces the robot, system counts shots automatically.
  - Copy this hybrid. Do not repeat the fully-autonomous attempt first.
- FRC Team 1294 Auto Scout.
  - Post: https://www.chiefdelphi.com/t/introducing-team-1294-s-auto-scout-program-the-p-a-c-k/516763
  - Code: https://github.com/packofparts/scouting-ai
  - Per-event corner calibration, Roboflow detector, Hungarian association, manual team numbers.
  - Explicitly does NOT detect points scored.
  - Users report path and start-position errors.
  - Copy: per-event calibration, manual team numbers.
  - Avoid: claiming scoring it does not do.
- Nomadic Dragons FTC 25528 Auto Judge.
  - Site: https://www.nomadic-dragons.com/
  - Claims "first-ever computer vision auto judge in FTC", real-time element counting.
  - No code, no demo, no accuracy numbers found. UNVERIFIED.
  - Do not treat it as proven. Do not copy its claims.

## Critical correction — game pieces

- BIOBUZZ 2026-27 game pieces are ~3-inch pollen balls.
- They are NOT purple/green artifacts.
- Purple/green artifacts are DECODE 2025-26.
- Pollen ball colors were not stated in the official preview.
- Verify colors from the BIOBUZZ game manual before labeling anything.
- Do not hardcode purple/green anywhere.
- Do not hardcode any color until verified.

## Training data

- BIOBUZZ footage only. DECODE footage excluded.
- 26 confirmed YouTube URLs, ~15 events, ~300+ matches.
- Full list in the appendix at the end of this brief.
- Best starter: FUN Robot in 30 Hours clipped matches.
  - Scores are in the video descriptions = free ground truth.
- Biggest chunk: Michigan FiM AV qualifiers (~200+ matches).
- Ground truth: official scores on ftc-events.firstinspires.org.
  - No hand-labeling needed for verification.
- Camera angle is UNVERIFIED for every video.
  - Watch a few minutes of each before training.
  - Confirm fixed full-field view.

## Holdout test set — do NOT train on these

- FTC STL North League Meet, Battle High School, Nov 14, 2026.
  - Stream: https://www.youtube.com/watch?v=EHLGE5Rl3qU
- Daisy League Meet 2, Newton IA, Nov 21, 2026.
  - FTC Iowa channel. URL not published yet.
- Michigan Midland Qualifier, Nov 20–21, 2026.
  - Or MI States, Dec 11–13, 2026.
- Alternative holdout: FIRST Chesapeake Washington DC, Dec 5–6, 2026.

## Identity — how to keep the right robot the right robot

- First: alliance color.
  - Red/blue bumper histogram. Robust. Halves the problem.
- Second: ReID + position to maintain identity across frames.
- Third: team-number OCR only as confirmation. It is fragile.
  - Vote OCR over top-20 crops offline (PlayerTV pattern).
  - Team 334 found a vision-language model beat classical OCR.
  - Iowa State got perfect counts in only 43% of matches.
- On uncertainty: flag, do not swap.
  - A wrong ID is worse than an unknown ID.
- Manual identity seeding and correction UI is a first-class feature.

## Scoring from video

- Primary signal: zone-crossing on top-down tracks.
  - Pickup → scoring = cycle.
- Confirmation: goal-ROI ball-count delta.
- Cross-check 1: scoreboard OCR (PaddleOCR, voted over frames).
- Cross-check 2: official ftc-events totals for the match.
- No FTC scoreboard-OCR evidence exists. Treat it as unproven until measured.
- Cycles are reliable. Per-robot point totals are the hardest part.
- Report cycles with confidence.
- Report points with their verification status.

## Labeling and training sizes

- Prototype: 50–100 images.
- Reliable: 200–300 instances per class.
- V1 target: 500–1,000 labeled frames (frames are instance-rich).
- Production target (Ultralytics guidance): 1,500 images / 10,000 instances per class.
- Free Colab T4 is enough for YOLO11n/s fine-tuning.
  - 12 h max session, ~90 min idle limit, unpublished quota.
  - Kaggle ~30 GPU-hours/week as backup.

## Control Scout link

- Video-derived per-team stats feed Scout team pages and the prediction engine.
  - Cycles, cycle time, zone time, speed.
- Auto-cropped robot photos from match frames go on Scout team pages.
- Timestamped highlights link from Scout match pages.
- Same versioned numbers everywhere.
  - Computation never forks, presentation stays free (Scout rule).

## Build order — four phases, in order, verify each before the next

- Phase 1: robot tracking + top-down trails. Fixed camera, 1–2 robots first.
- Phase 2: cycle counting via zone entries.
- Phase 3: highlight clipping with FFmpeg around detected events.
- Phase 4: scoring / element detection. Hardest. Optional.
- Verify each phase on the holdout set before moving on.
- Never claim an accuracy number without measuring it on the holdout.

## Repo and workflow

- New public repo under ZOGWORK.
  - Name: control-vision (confirm with Sushil if different).
- Every change via PR.
- Code-review bot must give 5/5 before merge. Never merge below 5/5.
- Commits authored as Sushil Muthuvelkumar (sushilm20@users.noreply.github.com).
- One feature per PR. Base main only. No stacked PRs.

## Open questions (do not guess — verify or ask Sushil)

- Pollen ball colors (check the BIOBUZZ game manual).
- FTC scoreboard OCR: does it work on real streams? Unproven.
- CPU fps figures are host-dependent. Measure on the target laptop.
- No published FTC end-to-end scoring accuracy exists.
  - Control Vision would be the first verified one.
- Deep HM-SORT 80.1 HOTA claim is disputed. Do not use it.

## Appendix — BIOBUZZ footage list (26 URLs)

- No videos were downloaded. Download with yt-dlp as needed.
- Ground truth for every event: official scores on ftc-events.firstinspires.org.

### Michigan qualifiers (FiM AV)

- Royal Oak, Nov 6–7, 28 teams.
  - Practice Day: https://www.youtube.com/watch?v=Jipe7rWPZWA
  - Day 1: https://www.youtube.com/watch?v=FGiUuys_-BE
- Belding, Nov 6–7, 27 teams, 34 quals.
  - Day 1: https://www.youtube.com/watch?v=RE0KNvjR07c
- Coloma, Nov 6–7, 31 teams.
  - Day 1: https://www.youtube.com/watch?v=ghEe-5fH_Kw
- Hartland, Nov 6–7, Howell, 30 teams, 38 quals.
  - Practice Day: https://www.youtube.com/watch?v=JkprQksmRIg
  - Day 1: https://www.youtube.com/watch?v=A5n0Xamahk8
- Bullock Creek, Nov 6–7, Midland, 34 teams, ~43 quals.
  - Practice Day: https://www.youtube.com/watch?v=2Y1fVAD-Lls
  - Day 1: https://www.youtube.com/watch?v=jXgqS-2PGvY
  - KILTS highlights only (not a full stream): https://www.youtube.com/watch?v=R1f0B7fM8g0

### Alabama

- Athens Qualifier, Oct 24, 2026.
  - Full-day stream: https://www.youtube.com/watch?v=4tScR8RexiA

### Arizona

- Quail Qualifier, Nov 7, Glendale, 25 teams, 32 quals + 10 playoffs.
  - https://www.youtube.com/watch?v=tLK-tRpZSTA
- High Country Qualifier, Nov 7, Flagstaff, 18 teams.
  - https://www.youtube.com/watch?v=ITK-Rc5LvT4

### California

- NorCal Screamage, Nov 1, San Jose.
  - https://www.youtube.com/watch?v=ffYioYR0TQE
- East Bay League Meet #1, Nov 1.
  - https://www.youtube.com/watch?v=44mne4zzkPU

### Iowa

- Daisy League Meet 1, Nov 7, Martensdale.
  - https://www.youtube.com/watch?v=EPhcvM19DnQ

### Ohio

- CSU & CMSD Scrimmage, Cleveland State / Cleveland Metro.
  - https://www.youtube.com/watch?v=s5-ACnmt5vk

### FUN Robot in 30 Hours — Wisconsin, Sep 13–14, 2026

- Full streams:
  - Saturday: https://www.youtube.com/watch?v=b79NPndUWKY
  - Sunday: https://www.youtube.com/watch?v=mnId-C15OTc
- Clipped matches (scores in descriptions):
  - Match 3: https://www.youtube.com/watch?v=CNaUmOvj2WQ (Red 21, Blue 41)
  - Match 5: https://www.youtube.com/watch?v=k-mSVM3Foos (Red 33, Blue 44)
  - Match 6: https://www.youtube.com/watch?v=xdIbYp4WX0U (Red 38, Blue 11)
  - Match 7: https://www.youtube.com/watch?v=zTEYuu9a0y4 (feed cut last 15s)
  - Match 9: https://www.youtube.com/watch?v=EUdBXBLy7No (Red 19, Blue 35)
  - Match 12: https://www.youtube.com/watch?v=AS1vn-KhBJs (Red 25, Blue 52)
- Georgia Tech Ri30H Sunday stream, Sep 13:
  - https://www.youtube.com/watch?v=2xiqGhEc7fc

### Single-robot demo — low training value

- 10653 Dragatrons 222-pt solo world record (not a match).
  - https://www.youtube.com/watch?v=kwYU2RU7V14
