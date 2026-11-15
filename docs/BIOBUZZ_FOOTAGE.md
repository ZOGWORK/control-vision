# FTC BIOBUZZ (2026–27) Match Videos on YouTube

- Training-set sweep, completed Tue Nov 10, 2026.
- Scope: BIOBUZZ only (2026–27 season).
- DECODE (2025–26) footage deliberately excluded.
- The initial URL sweep did not download footage. Later local testing downloaded five Ri30H match clips and one WA Salk Q3 clip; no footage or extracted frames are committed.

## Coverage at a glance

- **26 confirmed video URLs** across **~15 events**
- **~300+ matches** in the footage (rough estimate)
- Full-day livestreams: 16 (plus 1 team highlights reel)
- Clipped single matches: 6 (FUN Robot in 30 Hours)
- Official final scores are published on FIRST Events (ftc-events.firstinspires.org).
  They validate aggregate score comparisons only. Robot detections, identities,
  trajectories, and individual scoring cycles still require hand-labeled truth.

## Camera angle and local sample status

- The downloaded Ri30H and WA Salk Q3 samples were visually inspected as fixed cameras. The Ri30H angle has notable human/field occlusion; distant robots are small in the WA Salk view.
- WA Salk Q3 source: [FTC Quality Control match clip](https://www.youtube.com/watch?v=EFLYcMxrTOM); [official FIRST event result](https://ftc-events.firstinspires.org/2026/USWASAM1/qualifications) is red 180, blue 103.
- Local frame extraction produced 577 **unlabeled** frames across those six clips. They are not a training set until boxes are labeled and an event-separated split is prepared.
- All other listed URLs still need camera-angle screening.
- FiM AV (Michigan) qualifier streams are normally fixed full-field
  cameras, but that is the usual pattern, not a verified fact here.
- Confirm by watching a few minutes of each stream before training.

---

## Michigan qualifiers (FiM AV)

- Streamed through FiM AV hubs at fimav.us/live/2026/\<event-code\>.
- Each event has a Practice Day (Nov 6) and a Day 1 (Nov 7) video.
- 6 events, ~179 teams, ~200+ qualification matches + playoffs.

### Royal Oak Qualifier — Nov 6–7, 28 teams, code USMIROQ

- Practice Day (Nov 6):
  https://www.youtube.com/watch?v=Jipe7rWPZWA
- Day 1 (Nov 7):
  https://www.youtube.com/watch?v=FGiUuys_-BE

### Belding Qualifier — Nov 6–7, 27 teams, 34 quals

- Day 1 (Nov 7):
  https://www.youtube.com/watch?v=RE0KNvjR07c
- Practice Day: URL not found.

### Coloma Qualifier — Nov 6–7, 31 teams

- Day 1 (Nov 7):
  https://www.youtube.com/watch?v=ghEe-5fH_Kw
- Practice Day: URL not found.

### Hartland Qualifier — Nov 6–7, Howell, 30 teams, 38 quals, code USMIHOQ

- Practice Day (Nov 6):
  https://www.youtube.com/watch?v=JkprQksmRIg
- Day 1 (Nov 7):
  https://www.youtube.com/watch?v=A5n0Xamahk8

### Bullock Creek Qualifier — Nov 6–7, Midland, 34 teams, ~43 quals, code USMIMIQ1

- Practice Day (Nov 6):
  https://www.youtube.com/watch?v=2Y1fVAD-Lls
- Day 1 (Nov 7):
  https://www.youtube.com/watch?v=jXgqS-2PGvY
- Team KILTS highlights only (not a full stream):
  https://www.youtube.com/watch?v=R1f0B7fM8g0

### Southgate Qualifier — Nov 6–7, 29 teams

- No video URL found in multiple searches.
- FiM AV hub page exists (fimav.us/live/2026/\<code\>) but the
  event code was not resolved from search results.

---

## Alabama

### FTC Athens Qualifier — Oct 24, 2026, Athens Renaissance School

- Full-day livestream:
  https://www.youtube.com/watch?v=4tScR8RexiA

---

## Arizona

### Quail Qualifier — Nov 7, 2026, Glendale (Cactus HS), 25 teams

- 32 quals + 10 playoffs, code USAZGLQ1
- https://www.youtube.com/watch?v=tLK-tRpZSTA

### High Country Qualifier — Nov 7, 2026, Flagstaff (Coconino HS), 18 teams

- Code USAZFLQ, FIRST Tech Challenge Arizona channel
- https://www.youtube.com/watch?v=ITK-Rc5LvT4

---

## California

### NorCal BIOBUZZ Screamage — Nov 1, 2026, San Jose (The Play Space)

- Full stream of match play:
  https://www.youtube.com/watch?v=ffYioYR0TQE

### FTC East Bay League Meet #1 — Nov 1, 2026

- "7419 Tech Support Live Stream" (FRC Team 7419 hosting)
- https://www.youtube.com/watch?v=44mne4zzkPU

---

## Iowa

### Daisy League Meet 1 — Nov 7, 2026, Martensdale

- Replay available; FTC Iowa channel streams every Daisy League meet.
- https://www.youtube.com/watch?v=EPhcvM19DnQ

---

## Ohio

### FTC BioBuzz CSU & CMSD Scrimmage — Cleveland State / Cleveland Metro

- https://www.youtube.com/watch?v=s5-ACnmt5vk

---

## FUN Robot in 30 Hours — Wisconsin, Sep 13–14, 2026

- Clipped single matches; scores are in the video descriptions,
  so they are free ground truth.
- Full streams:
  - Saturday: https://www.youtube.com/watch?v=b79NPndUWKY
  - Sunday: https://www.youtube.com/watch?v=mnId-C15OTc
- Clipped matches:
  - Match 3: https://www.youtube.com/watch?v=CNaUmOvj2WQ
    (Red 21, Blue 41)
  - Match 5: https://www.youtube.com/watch?v=k-mSVM3Foos
    (Red 33, Blue 44)
  - Match 6: https://www.youtube.com/watch?v=xdIbYp4WX0U
    (Red 38, Blue 11)
  - Match 7: https://www.youtube.com/watch?v=zTEYuu9a0y4
    (feed cut in the last 15 sec)
  - Match 9: https://www.youtube.com/watch?v=EUdBXBLy7No
    (Red 19, Blue 35)
  - Match 12: https://www.youtube.com/watch?v=AS1vn-KhBJs
    (Red 25, Blue 52)

### Georgia Tech Ri30H Sunday stream — Sep 13, 2026

- Scrimmage matches:
  https://www.youtube.com/watch?v=2xiqGhEc7fc

---

## Single-robot demo — low training value

### BioBuzz Solo World Record, 10653 Dragatrons — 222 pts

- https://www.youtube.com/watch?v=kwYU2RU7V14
- Single-robot demo, not a match. List here for completeness only.

---

## Held-out test candidates (do NOT train on these)

### 1. FTC STL North League Meet — Battle High School, Nov 14, 2026

- Scheduled stream already published:
  https://www.youtube.com/watch?v=EHLGE5Rl3qU
- Why: future event, distinct venue (Missouri), different region
  from the Michigan-heavy training set.

### 2. Daisy League Meet 2 — Newton High School, Iowa, Nov 21, 2026

- Streams on the FTC Iowa channel (same channel as the Nov 7
  Martensdale meet in the training set, different venue/date).
- Why: tests generalization within the same camera setup.
- URL not yet published as of Nov 10; check the FTC Iowa channel
  after Nov 21.

### 3. Michigan Midland Qualifier — Nov 20–21, 2026 (or MI States, Dec 11–13)

- Same FiM AV pipeline as most of the training data.
- Why: tests temporal generalization — later-season robots and
  strategies against a model trained on early-season footage.
- Alternative: FIRST Chesapeake Washington DC, Dec 5–6, 2026
  (different region/provider entirely;
  streams at youtube.com/@FIRSTChesapeake/streams).

---

## BIOBUZZ events so far with NO footage found

- Southgate Qualifier, Michigan, Nov 6–7, 29 teams.
- Belding and Coloma Practice Day streams (Day 1 found for both).
- Don Bosco Prep Northeast League Meet, New Jersey, Nov 7, 2026
  (code USNJNEM2; Sushil mentored there).
- Marian Scrimmage, Omaha NE, Nov 7, 2026, 12 teams.
- Festival Regional SESI de Robótica – Pernambuco, Recife, Brazil,
  Oct 23–24, 2026, 12 teams.
- IQanat FIRST Regional, Burabay, Kazakhstan, Nov 7, 2026,
  21 teams, 32 quals + 10 playoffs (Instagram recaps only).
- Brisbane South Qualifier, Nov 7–8, 2026, 17 teams,
  26 quals + 7 playoffs.
- Victoria Qualifier, Caulfield East, Australia, Nov 7, 2026,
  10 teams, 15 quals + 2 playoffs.
- Robot in 7 Days Scrimmage, Sandy UT, Sep 19, 2026 (code USUTSAS1):
  video link points to ftccv.org, no direct YouTube URL found.
- DiscoverFest Scrimmage, Oklahoma City, Sep 26, 2026.
- Texas and Illinois league meets, Nov 6–8 (9 Texas + 7 Illinois
  meets): no individual YouTube streams found.
  Carver League Meet 1, Oct 31 (team 26949) was livestreamed via a
  QR-linked YouTube URL, but the URL itself was not recovered.
- Hawaii league meets stream on Twitch
  (twitch.tv/hawaiifirstrobotics), not YouTube.

## Upcoming events to watch for (as of Nov 10)

- Brazil: Rio Grande do Sul, Porto Alegre, Nov 10–11;
  Goiás, Goiânia, Nov 12–13; Paraná, Pinhais, Nov 12–13.
- Australia: South Australia Qualifier Nov 13–14;
  Brisbane North Qualifier Nov 13–14;
  Western Australia Qualifier Nov 14; South Bank Qualifier Nov 20.
- Michigan: Midland Qualifier Nov 20–21; MI States Dec 11–13.
- FIRST Chesapeake: Washington DC Dec 5–6; Powhatan Dec 5–6;
  Charlottesville/Manassas Dec 12–13.
- Iowa: Daisy League Meet 2, Nov 21 (FTC Iowa channel).

## Notes on method

- Searches used multiple phrasings plus event names from
  ftc-events.firstinspires.org and FTSee's Week 8 roundup
  (51 league meets Nov 6–8 across 13 US states + Canada;
  11 qualifiers that weekend worldwide).
- The four streams in an earlier test set (Alabama Athens,
  Royal Oak, Belding, Coloma) are BIOBUZZ, not DECODE —
  an earlier run mislabeled them.
- Don Bosco event code USNJNEM2 resolves on the 2026
  ftc-events path; the earlier "not found" used the 2025 path.
