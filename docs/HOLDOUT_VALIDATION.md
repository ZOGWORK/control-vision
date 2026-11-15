# Holdout accuracy evaluation

`python -m control_vision.validation.holdout` measures what the app saved in a
project database against human ground truth for held-out matches. It does not
create labels, does not check that labels are correct, and does not certify a
model or an app release.

**No real holdout ground truth exists yet, so no Control Vision accuracy number
has been measured.** The tests run the evaluator on small synthetic matches
only; their "pass" results show the arithmetic works, not that the app is
accurate. Do not quote any number from them.

## What the evaluator reads

From the project database (opened read-only), for one `video_id`:

- `videos`: path, duration, `analysis_status` (must be `ready`),
  `tracking_mode`, stored official scores, DQ flags and RP thresholds.
- `calibrations.corners_json`: the four fitted field corners.
- `robots` and `tracks`: per sampled frame, the field position (`x_ft`,
  `y_ft`) the app saved for each seeded team.
- `cycles`: counted per robot.
- `scoring_events`: the same rows as `ProjectDatabase.events_for_video`,
  including `status` (`confirmed`, `flagged`, `rejected`), `points`,
  `event_type`, robot and recipient alliance (`scoring_alliance`).
- `clips`: path, saved start and end per scoring event. Relative paths are
  resolved from the database folder.

Video points come from `control_vision.game.scoreboard.calculate_scoreboard`,
the same function the app and the Scout JSON export use.

## Ground-truth format (version 1)

One JSON file per holdout match. Make an unreviewed template with the sample
rate that produced the saved tracks:

```powershell
python -m control_vision.validation.holdout --make-template `
  --project path\to\control_vision.sqlite --video-id 1 --sample-fps 8 `
  --output labels\stl-north-q1.json
```

The template lists every seeded team at every sampled source frame. Fill it in:

```json
{
  "format_version": 1,
  "match_label": "STL North Q1",
  "source_match": "USMOSTLN:Q1",
  "project_db": "C:/path/to/control_vision.sqlite",
  "video_id": 1,
  "sample_fps": 8,
  "tracks": [
    {"frame": 0, "team_number": 1234, "visible": true, "x_ft": 2.1, "y_ft": 3.4},
    {"frame": 0, "team_number": 5678, "visible": true, "box_px": [410, 300, 470, 362]},
    {"frame": 0, "team_number": 9012, "visible": false, "x_ft": null, "y_ft": null}
  ],
  "cycles": [{"team_number": 1234, "count": 4}],
  "scoring_events": [{"t_s": 92.4, "team_number": 1234, "event_type": "hive_tip_teleop"}],
  "official_scores": {"red": 102, "blue": 94,
                      "source_url": "https://ftc-events.firstinspires.org/2026/USMOSTLN/qualifications/1"},
  "calibration_landmarks": [
    {"x_ft": 3, "y_ft": 3, "pixel_x": 420, "pixel_y": 480},
    {"x_ft": 9, "y_ft": 3, "pixel_x": 760, "pixel_y": 485},
    {"x_ft": 3, "y_ft": 9, "pixel_x": 415, "pixel_y": 830},
    {"x_ft": 9, "y_ft": 9, "pixel_x": 755, "pixel_y": 835}
  ]
}
```

(The track and cycle rows above are shortened; the real file must have every
seeded team.)

- `source_match` is `EVENT:MATCH`, written the same way as in the training
  `split_manifest.csv` (see `docs/DATASET_SPLITS.md`).
- `tracks`: one row per seeded team per sampled frame. `visible` must be
  reviewed `true` or `false`. A visible robot needs either its ground-contact
  field position `x_ft`/`y_ft` (0–12 ft) or a hand-drawn source-pixel box
  `box_px` `[x1, y1, x2, y2]`. A box is converted the same way the app converts
  detections: its bottom-center is projected through the saved calibration.
  Use the human-reviewed team identity. Do not mark a robot invisible because
  its number is hard to read.
- `cycles`: a hand-counted cycle total for every seeded team.
- `scoring_events`: every scored event, checked by a person against the video
  or clip, with `t_s` in seconds from the video start, `team_number` of the
  robot that scored (or `null` if it cannot be told), and optionally the app's
  `event_type`.
- `official_scores`: the official red and blue totals and the `https` link to
  the match on ftc-events.firstinspires.org that they were copied from. Other
  sources are rejected. If the project also stores official scores, they must
  agree. Per-robot points are never derived from these alliance totals.
- `calibration_landmarks`: at least four field points with known field
  coordinates, clicked on the calibration frame, that span an area and are at
  least 5 px from every fitted corner. The fitted corners always reproject
  almost exactly (see `reprojection_error_px` in
  `control_vision/pipeline/homography.py`), so they prove nothing.

## Metric definitions

Every per-match result keeps its raw counts and denominators. The summary
gives the match list, per-match rows, and totals.

- **Association.** In each labeled frame, visible truth robots are paired with
  the app's saved tracks within 1.5 ft on the field. As in CLEAR-MOT, a robot
  keeps the track it was last paired with while that track is still within
  1.5 ft. The rest are paired by Hungarian assignment, which first maximizes
  the number of pairs and then minimizes total distance.
- **ID switches.** The number of times a truth robot is paired with a
  different track from the one it was last paired with. A frame with no pair
  (fragmentation) does not count by itself; resuming on the same track is not
  a switch. Target: mean ≤ 2 per match.
- **Track completeness.** Paired truth observations divided by visible truth
  observations (robot frames labeled visible). Target ≥ 95%. The report also
  gives identity-aware completeness (paired with the track of the correct
  team), and the tracking phase also needs ≥ 95% identity-aware completeness
  for every robot. A robot that was never labeled visible counts as 0. This is
  stricter than the brief's wording, so a whole-match identity swap or one
  badly tracked robot cannot pass.
- **Reprojection error.** Pixel distance between each independent landmark and
  its field position projected through the saved homography. The maximum over
  all landmarks and matches must be < 5 px; RMS is reported too.
- **Cycles.** A match passes when every team's app cycle count is within ±1 of
  the hand count. Target: ≥ 90% of matches.
- **Highlights.** Every non-rejected app scoring event needs a decodable clip
  whose saved start and end are within 0.15 s of the event time −3 s and +3 s,
  clamped to the video, and whose decoded length is within 0.5 s of that
  window. Every human-labeled scoring event must also be paired (within 1 s,
  same `event_type` when given, Hungarian assignment) with an app event that
  has such a clip. Target: 100% of both.
- **Score.** A match passes when its review queue has no flagged events and
  each alliance's video score is within ±10% of its official score (an
  official 0 must be matched exactly). Alliances are checked separately so
  errors in opposite directions cannot cancel. Target: ≥ 80% of matches.
- **Robot attribution.** Confirmed app events with nonzero points are paired
  with human-labeled events (within 1 s, same `event_type` when given). The
  numerator is human events with a known robot whose paired app event credits
  the same team. The denominator is human events with a known robot plus app
  scored events that pair with no human event. Missed events and false
  positives both count against it. Target ≥ 80%.

## Status values

Each phase is `pass`, `fail`, `insufficient_matches`, or `not_measured`:

- `not_measured`: no data for that metric, for example no scoring events in
  any match. This is never treated as a pass.
- `insufficient_matches`: the metric meets its target but on fewer than five
  matches (for highlights and attribution, matches that have the needed
  events). The brief needs at least five hand-counted holdout matches.
- `fail`: the metric misses its target on any number of matches.

`acceptance_targets_met` is true only when all four phases pass, at least
five distinct matches were evaluated, no result came from the experimental
`seeded_optical_flow` preview, holdout/training disjointness was verified,
and the holdout set came from a frozen, hashed list. Otherwise `blockers`
says why.

## Holdout set integrity

The build brief names STL North (November 14, 2026), Daisy Meet 2
(November 21), and Michigan Midland (November 20–21) as holdout sources.

- **Frozen list.** Write the list of truth files before evaluating and do not
  change it afterwards:

  ```json
  {"cases": [{"path": "labels/stl-north-q1.json", "sha256": "…"}]}
  ```

  `--holdout-list` evaluates exactly those files and refuses any whose SHA-256
  changed.
- **Disjoint from training.** `--dataset-root` (and optionally `--manifest`)
  runs the full `split_manifest` audit on the training dataset and rejects any
  holdout match whose event appears in its train or val rows. This is the
  event-level rule from `docs/DATASET_SPLITS.md`: no frame from a holdout
  event may be used for training, even from another match. It cannot prove the
  deployed weights were trained on that dataset; keep the training record
  with the report.
- Each match, source video and truth file can be counted only once.

```powershell
python -m control_vision.validation.holdout --holdout-list labels\holdout.json `
  --dataset-root datasets\biobuzz-v1 --output reports\holdout.json
```

## Reproducibility

Each case records the SHA-256 of its truth file, project database, source
video and every decodable clip. The report records the evaluator settings
(gate, tolerances, truth format version, Control Vision version) and the
frozen list and split manifest hashes. Re-running on the same files gives the
same report.

## Not measured here

- The CPU budget (a 2:30 match at 8 fps processed in ≤ 10 minutes on the
  reference laptop). The database does not record processing time; the
  report always shows it as `not_measured`. Time it on the target laptop.
- Whether labels are correct, whether clips show the right action, and whether
  the packaged app passed release checks.
- Real accuracy: no holdout match has been labeled yet.
