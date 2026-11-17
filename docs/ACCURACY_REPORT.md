# Control Vision accuracy report

This is the only place an accuracy figure for Control Vision may be quoted from (see
`docs/CONTROL_VISION_UI_SKILL.md`, rule 6). Every number here names the matches it was measured on and how. A number
that is not in this file has not been measured.

Last updated 2026-11-17, against `main` after #37 and #38.

## Holdout events

These events are never used for training, validation, model selection or rule tuning:

| Event | Code | Footage on disk |
|---|---|---|
| High Country – Flagstaff Qualifier | USAZFLQ | Q5, Q8, Q16, Q20 |
| Salk League Meet 1 | USWASAM1 | Q3 |
| STL North League Meet | — | none |
| Daisy League Meet 2 | — | none |
| USMIMIQ2 | USMIMIQ2 | none |
| MI States | — | none |

Codes shown as — are not recorded here and are not guessed. Only High Country and Salk have footage, so every
holdout figure below comes from those two events.

**Ground-truth limits:**
- Detector boxes on these frames were proposed by a model and then reviewed by a person. They are not a hand-drawn
  gold set.
- There is no labelled per-robot truth (positions, identities, cycles) for any holdout match. The evaluator in
  `docs/HOLDOUT_VALIDATION.md` has no truth files yet.
- Official FTC results are per alliance, so no official per-robot score exists.

## 1. Detector (v0.4.0, YOLO11n, 4 classes)

At the app's operating point (confidence 0.25, NMS IoU 0.5, a match at IoU ≥ 0.5), measured 2026-11-16:

| Set | Images | Robot precision | Robot recall |
|---|---|---|---|
| High Country holdout frames | 180 | 0.963 | 0.912 |
| High Country shot windows | 397 | 0.933 | 0.887 |
| Salk Q3 | 86 | 0.925 | 0.884 |

Threshold-free mAP50 (Ultralytics val) on the same sets:

| Set | Robot | Pollen | Red nectar | Blue nectar |
|---|---|---|---|---|
| High Country holdout frames | 0.948 | 0.806 | 0.984 | 0.896 |
| High Country shot windows | 0.905 | 0.790 | 0.480 | 0.714 |
| Salk Q3 | 0.944 | 0.948 | 0.983 | 0.988 |

**Weak spots:**
- Red nectar on High Country shot windows (mAP50 0.480) is the weakest class.
- Nectar precision on the High Country frames is about 0.39 at confidence 0.25 for both v0.3.0 and v0.4.0, while recall
  is about 0.96. That suggests unlabelled nectar in those frames rather than model error, but it has not been
  verified.
- A stage-2 fine-tune (v0.4.1 candidate) was rejected: robot recall fell 1–2 points on all three holdout sets.

## 2. Reading the live score overlay (scoring events per alliance)

`control_vision/pipeline/live_score.py` reads the scorekeeper's FTC Live overlay. It doesn't detect scoring from the
robots. Official values are from FTCScout's public API (fetched 2026-11-15). The full method is in
`docs/LIVE_SCORE_ACCURACY.md`.

| Matches | Alliance results | AUTO HIVE TIPs exact | TELEOP HIVE TIPs exact | Foul credit exact |
|---|---|---|---|---|
| High Country Q5, Q8, Q16, Q20 (holdout) | 8 | 8 / 8 | 8 / 8 | 8 / 8 |
| All 29 readable qualification clips, 9 events (not a holdout set) | 58 | 58 / 58 | 58 / 58 | 54 / 58 |

**Limits:**
- The overlay never shows the final total. PARK and end-of-match counts appear only on the results screen.
- The overlay doesn't show which robot scored.
- A HIVE TIP is a HIVE state change, not one robot's action, so it can't be attributed to a single robot.

## 3. End to end: per-robot tracking on the High Country holdout matches

Settings: v0.4.0 ONNX, 8 samples/s, confidence 0.25. Seeds were the robots' start positions picked by eye, with no
reviewer corrections. Coverage is the share of sampled in-play frames with a recorded position for that robot.
Full table: `HOLDOUT_E2E_REPORT` (local, not in the repo, because it references private footage).

| Measured on | Robots | Mean coverage | Median | Robots above 75% |
|---|---|---|---|---|
| Q5, Q8, Q16, Q20; `main` after #38 | 15 seeded (+1 not visible at the start) | 43.8% of seeded robots | 33% | 1 of 16 |

- **#36 and #37 on this holdout:** the colour rule (#36) and the 20 s re-link window (#37) changed one robot's coverage
  by 0.5 points (Q8) and nothing else.
- **Development matches:** on train/val matches, the same changes raised RO3 from 19.2% to 30.7% and HL10 from 26.5%
  to 33.7%. Those are development numbers, not accuracy.
- **Identity:** a by-eye spot check of 44 frames, every 15 s across the four matches, plus all 5 automatic re-links found
  no robot carrying another robot's identity. A robot that can't be followed is left as lost, not guessed. This is a
  spot check, not a measured error rate.
- **One reviewer correction per lost robot at 60 s** (simulated): mean coverage went from 43.8% to 47.4%.
  Corrected robots were lost again after 1.5–40 s.

## Not measured yet

- Per-robot position error, cycle counts and per-robot scoring. These need labelled truth for a holdout match.
- Any holdout figure for STL North, Daisy League Meet 2, USMIMIQ2 or MI States. There's no footage for them.
- Final match totals from the results-screen OCR (`scoreboard_ocr.py`) against official totals on holdout matches.
