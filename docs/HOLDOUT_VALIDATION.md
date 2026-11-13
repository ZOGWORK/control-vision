# Holdout accuracy evaluation

This command measures saved Control Vision results against human labels. It does
not create labels, establish that a model was trained correctly, or certify an
app release. Use only matches excluded from training, record their event and
match names, and keep the original video with the project database.

The build brief names STL North (November 14, 2026), Daisy Meet 2 (November
21), and Michigan Midland (November 20–21) as the initial holdout sources.
Do not train on those matches. The required five or more labeled holdout
matches must be distinct videos.

## Make a label template

Use the same sample rate that produced the saved tracks:

```powershell
python -m control_vision.validation.holdout --make-template `
  --project path\to\control_vision.sqlite --video-id 1 --sample-fps 8 `
  --output labels\stl-north-q1.json
```

The template lists **every seeded team at every sampled source frame**. Review
the video and change each `visible: null` to `true` or `false`. For visible
robots, enter the ground-contact position as `x_ft` and `y_ft` in the calibrated
12 × 12 foot field. Leave invisible positions null. Do not mark a visible
robot invisible because its identity is hard to read; use the human-reviewed
team identity and note uncertainty outside the JSON if necessary. The evaluator
rejects unreviewed visibility flags and missing frame/team entries.

Fill in `match_label`, the hand-counted cycle `count` for every team, the
official red and blue totals, and a human-reviewed `scoring_events` list. Each
scored event needs `t_s` (seconds from the video start), `team_number`, and,
when known, `event_type` using the app's scoring category. Include events for
which the robot is unknown with `team_number: null`; these cannot count as
correct robot attribution. Label at least four **independent** field landmarks
on the calibration frame with known `x_ft`, `y_ft`, `pixel_x`, and `pixel_y`.
Do not reuse the four corners clicked to fit the homography; that would make
reprojection error look artificially perfect.

The populated JSON has this shape (the two shown track rows are illustrative;
the real file must contain every team at every sampled frame):

```json
{
  "match_label": "STL North Q1",
  "project_db": "C:/path/to/control_vision.sqlite",
  "video_id": 1,
  "sample_fps": 8,
  "tracks": [
    {"frame": 0, "team_number": 1234, "visible": true, "x_ft": 2.1, "y_ft": 3.4},
    {"frame": 0, "team_number": 5678, "visible": false, "x_ft": null, "y_ft": null}
  ],
  "cycles": [
    {"team_number": 1234, "count": 4},
    {"team_number": 5678, "count": 3}
  ],
  "scoring_events": [
    {"t_s": 92.4, "team_number": 1234, "event_type": "hive_tip_teleop"}
  ],
  "official_scores": {"red": 102, "blue": 94},
  "calibration_landmarks": [
    {"x_ft": 3, "y_ft": 3, "pixel_x": 420, "pixel_y": 480}
  ]
}
```

## Evaluate

```powershell
python -m control_vision.validation.holdout labels\stl-q1.json labels\stl-q2.json `
  labels\daisy-q1.json labels\midland-q1.json labels\midland-q2.json `
  --output reports\holdout.json
```

The report names every match and includes raw denominators. Track completeness
counts visible labeled observations with a prediction within 1.5 feet, using
one-to-one matching per frame. ID switches count changes in the matched
predicted team for a real robot. A separate identity-correct fraction catches
robots that are assigned the wrong team for the whole match, even when the ID
switch count is zero. Calibration reprojection error uses independent
landmarks in pixels.

The aggregate checks the build brief's targets: mean ID switches ≤ 2 per
match, track completeness ≥ 95%, maximum calibration error < 5 px, every
team's cycle count within ±1 in ≥ 90% of matches, every non-rejected scoring
event covered by a decodable clip from 3 seconds before through 3 seconds
after (within video bounds), combined video score within 10% of official total
in ≥ 80% of matches,
and correct robot attribution on ≥ 80% of confirmed scored events. A match
with pending flagged evidence fails its score check. `brief_targets_met` also
requires five distinct matches and excludes the experimental optical-flow
preview mode.

The report alone cannot verify that labels are correct, that the footage was
excluded from training, or that the trained weights and packaged app passed
release checks. Keep the labeled files and source-video list with the report.
