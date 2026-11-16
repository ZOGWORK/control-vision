# Live score overlay: measured accuracy

`control_vision/pipeline/live_score.py` reads the FTC Live overlay that event streams show during a match (each alliance's running score, the match clock and each alliance's HIVE TIP counter) and proposes timestamped scoring events. Every event is flagged for human review; nothing is confirmed automatically.

## What was measured

- **Footage:** the 32 qualification clips in the local footage set (9 events in Michigan, Alabama, Arizona and Iowa, Nov 2026), from public YouTube streams. The clips are not distributed.
- **Ground truth:** official per-match score breakdowns from FTCScout's public API (`autoHiveTips`, `dcHiveTips`, `penaltyPointsByOpp`, `totalPoints`), fetched Nov 15, 2026.
- **Method:** `python scripts/eval_live_score.py read …` reads each clip at 2 samples/s; `… report …` recomputes the proposals with the current code and compares per alliance. A field counts as exact only if it equals the official value.
- **Denominator:** 29 matches had a readable overlay = 58 alliance results. The 3 Quail Creek clips (USAZGLQ1) were not compared: that stream does not show the standard overlay, so it was not found.

## Results

Alliance results compared: 58

- auto tips: 58 / 58 exact
- teleop tips: 58 / 58 exact
- foul credit: 54 / 58 exact

Not compared:

- az-quail-Q13: Could not find the live score overlay in az-quail-Q13.mp4. The video needs the FTC Live overlay with the "Red" and "Blue" alliance labels.
- az-quail-Q19: Could not find the live score overlay in az-quail-Q19.mp4. The video needs the FTC Live overlay with the "Red" and "Blue" alliance labels.
- az-quail-Q25: Could not find the live score overlay in az-quail-Q25.mp4. The video needs the FTC Live overlay with the "Red" and "Blue" alliance labels.


### The four foul-credit misses

- **USMIHOQ Q25 blue, USMIROQ Q21 blue, USMIROQ Q29 red:** the official result has a 20-point foul credit that never appears on the overlay within the clip. The clips end a few seconds after the match, so the credit may have been entered later; this was not verified.
- **USMIBEQ Q4 blue:** a 5-point rise during TELEOP without a TIP was proposed as a MINOR FOUL credit; officially there was none. A 5-point rise is ambiguous on the overlay alone.

### What the overlay does not give

- **Final totals.** The last live score is below the official total in every match, because PARK and the end-of-match CELL, FLOWER and GARDEN counts are assessed after the match and appear on the results screen, not on the live overlay. The final total comes from the existing results-screen OCR (`scoreboard_ocr.py`).
- **Which robot scored.** The overlay gives the alliance and the time, not the robot. Attributing an event to a robot needs tracking around that time and stays a reviewer decision.
- **Ground truth for vision.** These numbers measure reading the scorekeeper's live entries. They are not a measurement of the detector or of automatic event detection from video.

## Per-match results

| Match | Alliance | Live: AUTO TIPs / TELEOP TIPs / foul credit | Official | Last live score | Official total |
|---|---|---|---|---|---|
| USALATQ Q11 | red | 0 / 0 / 5 | 0 / 0 / 5 | 8 | 25 |
| USALATQ Q11 | blue | 0 / 1 / 20 | 0 / 1 / 20 | 43 | 59 |
| USALATQ Q2 | red | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 8 |
| USALATQ Q2 | blue | 0 / 1 / 5 | 0 / 1 / 5 | 28 | 37 |
| USALATQ Q4 | red | 0 / 0 / 0 | 0 / 0 / 0 | 3 | 21 |
| USALATQ Q4 | blue | 0 / 1 / 0 | 0 / 1 / 0 | 1 | 35 |
| USAZFLQ Q16 | red | 0 / 1 / 0 | 0 / 1 / 0 | 23 | 36 |
| USAZFLQ Q16 | blue | 0 / 0 / 0 | 0 / 0 / 0 | 8 | 21 |
| USAZFLQ Q20 | red | 0 / 0 / 0 | 0 / 0 / 0 | 13 | 27 |
| USAZFLQ Q20 | blue | 0 / 2 / 0 | 0 / 2 / 0 | 56 | 66 |
| USAZFLQ Q5 | red | 0 / 0 / 0 | 0 / 0 / 0 | 13 | 34 |
| USAZFLQ Q5 | blue | 0 / 0 / 0 | 0 / 0 / 0 | 3 | 3 |
| USAZFLQ Q8 | red | 0 / 1 / 0 | 0 / 1 / 0 | 23 | 37 |
| USAZFLQ Q8 | blue | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 20 |
| USIADAIM1 Q16 | red | 1 / 0 / 0 | 1 / 0 / 0 | 31 | 50 |
| USIADAIM1 Q16 | blue | 0 / 0 / 0 | 0 / 0 / 0 | 3 | 17 |
| USIADAIM1 Q33 | red | 0 / 1 / 0 | 0 / 1 / 0 | 31 | 47 |
| USIADAIM1 Q33 | blue | 0 / 1 / 0 | 0 / 1 / 0 | 36 | 41 |
| USIADAIM1 Q9 | red | 1 / 0 / 0 | 1 / 0 / 0 | 31 | 54 |
| USIADAIM1 Q9 | blue | 1 / 2 / 0 | 1 / 2 / 0 | 71 | 78 |
| USMIBEQ Q24 | red | 0 / 1 / 0 | 0 / 1 / 0 | 36 | 53 |
| USMIBEQ Q24 | blue | 0 / 0 / 0 | 0 / 0 / 0 | 3 | 25 |
| USMIBEQ Q4 | red | 1 / 0 / 0 | 1 / 0 / 0 | 23 | 38 |
| USMIBEQ Q4 | blue | 0 / 0 / 5 ✗ | 0 / 0 / 0 | 13 | 25 |
| USMIBEQ Q7 | red | 1 / 2 / 0 | 1 / 2 / 0 | 76 | 98 |
| USMIBEQ Q7 | blue | 0 / 1 / 0 | 0 / 1 / 0 | 31 | 41 |
| USMIMIQ1 Q22 | red | 1 / 3 / 0 | 1 / 3 / 0 | 96 | 114 |
| USMIMIQ1 Q22 | blue | 1 / 1 / 0 | 1 / 1 / 0 | 56 | 81 |
| USMIMIQ1 Q34 | red | 0 / 1 / 0 | 0 / 1 / 0 | 28 | 44 |
| USMIMIQ1 Q34 | blue | 0 / 1 / 0 | 0 / 1 / 0 | 36 | 55 |
| USMIMIQ1 Q42 | red | 1 / 2 / 0 | 1 / 2 / 0 | 76 | 99 |
| USMIMIQ1 Q42 | blue | 1 / 2 / 0 | 1 / 2 / 0 | 76 | 95 |
| USMIMIQ1 Q6 | red | 1 / 0 / 20 | 1 / 0 / 20 | 46 | 57 |
| USMIMIQ1 Q6 | blue | 1 / 1 / 20 | 1 / 1 / 20 | 68 | 86 |
| USMICOQ Q23 | red | 1 / 2 / 20 | 1 / 2 / 20 | 91 | 101 |
| USMICOQ Q23 | blue | 0 / 0 / 0 | 0 / 0 / 0 | 11 | 23 |
| USMICOQ Q3 | red | 1 / 1 / 0 | 1 / 1 / 0 | 56 | 62 |
| USMICOQ Q3 | blue | 0 / 1 / 0 | 0 / 1 / 0 | 28 | 49 |
| USMICOQ Q37 | red | 0 / 1 / 0 | 0 / 1 / 0 | 31 | 53 |
| USMICOQ Q37 | blue | 0 / 1 / 0 | 0 / 1 / 0 | 25 | 34 |
| USMICOQ Q6 | red | 0 / 1 / 0 | 0 / 1 / 0 | 31 | 36 |
| USMICOQ Q6 | blue | 0 / 1 / 0 | 0 / 1 / 0 | 28 | 38 |
| USMIHOQ Q10 | red | 0 / 1 / 0 | 0 / 1 / 0 | 25 | 44 |
| USMIHOQ Q10 | blue | 2 / 3 / 0 | 2 / 3 / 0 | 106 | 127 |
| USMIHOQ Q25 | red | 2 / 3 / 0 | 2 / 3 / 0 | 108 | 116 |
| USMIHOQ Q25 | blue | 1 / 3 / 0 ✗ | 1 / 3 / 20 | 96 | 127 |
| USMIHOQ Q27 | red | 0 / 1 / 0 | 0 / 1 / 0 | 36 | 41 |
| USMIHOQ Q27 | blue | 0 / 3 / 0 | 0 / 3 / 0 | 76 | 83 |
| USMIHOQ Q5 | red | 0 / 0 / 0 | 0 / 0 / 0 | 8 | 19 |
| USMIHOQ Q5 | blue | 1 / 5 / 0 | 1 / 5 / 0 | 123 | 136 |
| USMIROQ Q21 | red | 1 / 2 / 0 | 1 / 2 / 0 | 68 | 94 |
| USMIROQ Q21 | blue | 0 / 0 / 0 ✗ | 0 / 0 / 20 | 13 | 48 |
| USMIROQ Q29 | red | 0 / 2 / 5 ✗ | 0 / 2 / 25 | 53 | 89 |
| USMIROQ Q29 | blue | 0 / 0 / 0 | 0 / 0 / 0 | 8 | 26 |
| USMIROQ Q3 | red | 1 / 7 / 0 | 1 / 7 / 0 | 163 | 181 |
| USMIROQ Q3 | blue | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 6 |
| USMIROQ Q7 | red | 0 / 3 / 0 | 0 / 3 / 0 | 63 | 77 |
| USMIROQ Q7 | blue | 1 / 2 / 0 | 1 / 2 / 0 | 65 | 89 |
