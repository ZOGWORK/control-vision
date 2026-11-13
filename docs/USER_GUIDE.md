# Control Vision user guide

## Prepare a local install

Use Python 3.11 or newer, install the pinned packages in `requirements.txt`, and run `python -m control_vision`. A locally trained YOLO11n ONNX file with class 0 named `robot` is required for detection. Control Vision intentionally does not download weights. Install FFmpeg and add `ffmpeg.exe` to PATH to make highlight clips; the full path can also be entered under Settings.

## Analyze a practice video or match

1. Create a project in a folder or reopen its `control_vision.sqlite` file.
2. Import an MP4, MOV, or MKV. Enter the optional event code and match number. Low resolution and low frame rate produce warnings but do not block import.
3. Choose a clear field frame and click the field corners clockwise, starting at the top-left. The points map to a 12 by 12 foot square. Reuse Calibration copies a previous video calibration and scales the corner pixels to the new resolution; visually confirm it before processing.
4. On the top-down map, choose a zone type and Draw Zone. Click polygon vertices and double-click to finish. Use names that match the field manual. No default zone coordinates are supplied because the manual diagram was not included.
5. Seed each visible robot by clicking near its floor contact point, then enter the team number and alliance. Up to four robots are supported.
6. Set the local model path, confidence threshold, and sample rate under Settings, then press Process. Cancel stops after the current frame. Logs go to `run_logs/`.

## Interpret results

The trail view follows the playable video timeline. Cycles are pickup-to-scoring zone trips and carry medium confidence because the current pipeline does not detect game pieces. Each scoring-zone trip creates a zero-point flagged candidate. You can also add a candidate at the current video time for events such as a HIVE tip or PARK. In Review Queue, select the event type, adjust the point count, set confidence, confirm/reject it, and reassign the robot if needed. The event-type presets follow BIOBUZZ Competition Manual TU09; visual evidence still needs human interpretation. Only confirmed points contribute to video totals. Enter the official red and blue alliance totals to compare them. The match remains labeled unreviewed while any flagged events remain.

Generate highlight clips after processing. Clips cover up to three seconds before and after each non-rejected candidate; boundary clips are shorter at the start or end of the video. Double-click a review row to preview its clip, or jump to the source video when no clip exists.

Statistics CSV and Scout JSON exports use the reviewed database values. Official scores are alliance totals, so the Scout team's `points_official` remains null rather than repeating an alliance value for every robot. Robot PNGs use the largest tracked crop, with confidence as the tie-breaker and a 20% margin. Crops smaller than 200 by 200 pixels are skipped and logged.

## Known verification limits

No trained weights, labeled dataset, or holdout videos ship with the source. The UI and code do not claim measured ID-switch, track-completeness, cycle-count, scoring, or CPU-performance results. Zone crossings and manually reviewed points are practical scaffolding; score-value inference from robot motion alone is not automatic.
