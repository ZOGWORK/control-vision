"""Live score overlay: parsing, layout, counter finding and event proposals."""
import unittest

import cv2
import numpy as np

from control_vision.pipeline.live_score import (
    AUTO, ENDED, TELEOP, TRANSITION, counter_box, layout_from_labels, match_phases, parse_clock,
    parse_score, parse_tips, propose_events, read_frame, stable_steps, summarize)


def sample(t_s, red=None, blue=None, red_tips=None, blue_tips=None, clock=None):
    return {"t_s": t_s, "red_score": red, "blue_score": blue, "red_tips": red_tips,
            "blue_tips": blue_tips, "clock_s": clock}


def timeline(points, step=0.5):
    """Samples from (t_s, overrides) keyframes, holding values between them."""
    state = {"red": 0, "blue": 0, "red_tips": 0, "blue_tips": 0, "clock": 150}
    rows = []
    keys = dict(points)
    t = 0.0
    end = max(keys) + 2.0
    while t <= end:
        for when, change in points:
            if abs(when - t) < 1e-9:
                state.update(change)
        rows.append(sample(round(t, 3), state["red"], state["blue"], state["red_tips"], state["blue_tips"], state["clock"]))
        t += step
    return rows


# Royal Oak Q3 (USMIROQ), simplified: an AUTO TIP, a LEAVE entered and two
# scorekeeper corrections at the end of AUTO, TELEOP TIPS, then the end.
ROYAL_OAK = [
    (16.5, {"red": 20, "red_tips": 1, "clock": 138}),
    (32.5, {"red": 28, "clock": 126}),
    (33.5, {"red": 31}),
    (34.5, {"red": 26}),
    (36.5, {"clock": 120}),
    (38.0, {"clock": 7}),
    (41.0, {"red": 23}),
    (46.0, {"clock": 119}),
    (60.0, {"red": 43, "red_tips": 2, "clock": 106}),
    (78.5, {"red": 63, "red_tips": 3, "clock": 88}),
    (166.0, {"clock": 0}),
]


class ParsingTests(unittest.TestCase):
    def test_clock(self):
        self.assertEqual(parse_clock("2:30"), 150)
        self.assertEqual(parse_clock(" 0:07"), 7)
        self.assertEqual(parse_clock("1.45"), 105)
        self.assertIsNone(parse_clock("1:75"))
        self.assertIsNone(parse_clock("9:00"))
        self.assertIsNone(parse_clock(""))

    def test_score_reads_round_zero_symbols_and_rejects_implausible_values(self):
        self.assertEqual(parse_score(" 43 "), 43)
        self.assertEqual(parse_score("。"), 0)
        self.assertEqual(parse_score("O"), 0)
        self.assertIsNone(parse_score("037"))
        self.assertIsNone(parse_score("1234"))
        self.assertIsNone(parse_score("X"))

    def test_tips(self):
        self.assertEqual(parse_tips("2 "), 2)
        self.assertEqual(parse_tips("。"), 0)
        self.assertEqual(parse_tips("10"), 10)
        self.assertIsNone(parse_tips("X"))
        self.assertIsNone(parse_tips("123"))


class LayoutTests(unittest.TestCase):
    def test_layout_follows_labels_at_top_and_bottom(self):
        top = layout_from_labels((719, 111, 778, 141), (1141, 108, 1207, 143), (1080, 1920, 3))
        bottom = layout_from_labels((718, 935, 779, 968), (1141, 933, 1208, 969), (1080, 1920, 3))
        # Score digits measured on the two streams lie inside the score areas.
        for layout, red_digits, blue_digits in ((top, (692, 160, 804, 242), (1148, 165, 1198, 236)),
                                                (bottom, (697, 987, 803, 1067), (1145, 988, 1201, 1063))):
            self.assertIsNotNone(layout)
            for area, digits in ((layout.red_score, red_digits), (layout.blue_score, blue_digits)):
                self.assertLessEqual(area[0], digits[0]); self.assertLessEqual(area[1], digits[1])
                self.assertGreaterEqual(area[2], digits[2]); self.assertGreaterEqual(area[3], min(digits[3], 1080))

    def test_labels_out_of_order_or_on_different_rows_are_rejected(self):
        self.assertIsNone(layout_from_labels((1141, 108, 1207, 143), (719, 111, 778, 141), (1080, 1920, 3)))
        self.assertIsNone(layout_from_labels((719, 111, 778, 141), (1141, 400, 1207, 435), (1080, 1920, 3)))


def band_with_counter(digit_box=True, checkmark_panel=False):
    band = np.zeros((100, 380, 3), np.uint8)
    band[:] = (30, 30, 200)  # red overlay background (BGR)
    if checkmark_panel:
        band[10:90, 20:110] = 255
        cv2.line(band, (35, 25), (95, 75), (0, 0, 0), 4)
    if digit_box:
        band[25:75, 250:290] = 255
        cv2.putText(band, "2", (258, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(band, "/4", (300, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return band


class CounterBoxTests(unittest.TestCase):
    def test_finds_white_box_with_dark_digit_not_white_text(self):
        box = counter_box(band_with_counter(), "right")
        self.assertEqual(box, (250, 25, 290, 75))

    def test_prefers_the_box_nearest_the_team_numbers(self):
        self.assertEqual(counter_box(band_with_counter(checkmark_panel=True), "right"), (250, 25, 290, 75))
        self.assertEqual(counter_box(band_with_counter(checkmark_panel=True), "left"), (20, 10, 110, 90))

    def test_no_counter_box(self):
        self.assertIsNone(counter_box(band_with_counter(digit_box=False), "right"))
        self.assertIsNone(counter_box(np.zeros((0, 0, 3), np.uint8), "right"))

    def test_read_frame_marks_a_missing_counter_as_unread(self):
        frame = np.zeros((300, 1920, 3), np.uint8)
        layout = layout_from_labels((719, 111, 778, 141), (1141, 108, 1207, 143), (300, 1920, 3))
        seen = []

        def recognise(crops):
            # Any real crop reads as a number, so only the blank placeholder
            # for a missing counter box can give an unread TIP count.
            seen.extend(crops)
            return [("" if crop.shape[:2] == (8, 8) and crop.min() == 255 else "7", 0.9) for crop in crops], 0.0

        row = read_frame(recognise, frame, layout)
        self.assertEqual((row["red_score"], row["red_tips"], row["blue_tips"]), (7, None, None))
        self.assertEqual([crop.shape[:2] for crop in seen[2:4]], [(8, 8), (8, 8)])


class StableStepTests(unittest.TestCase):
    def test_single_misreads_and_unreadable_samples_are_ignored(self):
        rows = [sample(0, red=0), sample(0.5, red=0), sample(1, red=8), sample(1.5, red=None),
                sample(2, red=20), sample(2.5, red=None), sample(3, red=20)]
        self.assertEqual(stable_steps(rows, "red_score"), [(0.0, 0), (2.0, 20)])

    def test_phases(self):
        phases = match_phases(timeline(ROYAL_OAK))
        self.assertEqual([phase for _, phase in phases], [AUTO, TRANSITION, TELEOP, ENDED])
        self.assertEqual([t for t, _ in phases], [0.0, 38.0, 46.0, 166.0])


class ProposalTests(unittest.TestCase):
    def test_royal_oak_q3_tips_and_corrections(self):
        proposal = propose_events(timeline(ROYAL_OAK))
        kinds = [(event["t_s"], event["event_type"], event["points"]) for event in proposal["events"]]
        # 20 -> 28 -> 31 -> 26 -> 23: the two corrections leave one +3 (LEAVE).
        self.assertEqual(kinds, [(16.5, "hive_tip_auto", 20), (32.5, "unclassified", 3),
                                 (60.0, "hive_tip_teleop", 20), (78.5, "hive_tip_teleop", 20)])
        self.assertTrue(all(event["status"] == "flagged" for event in proposal["events"]))
        self.assertEqual(summarize(proposal)["red"]["event_points"], 63)
        self.assertEqual(summarize(proposal)["red"]["auto_tips"], 1)
        self.assertIn("reduced", proposal["events"][1]["note"])

    def test_correction_cancels_several_rises_that_add_up(self):
        # Belding Q4: three TIPS entered in AUTO, LEAVE added, then two TIPS taken back.
        rows = timeline([(20.0, {"red": 20, "red_tips": 1}), (21.0, {"red": 40, "red_tips": 2}),
                         (22.5, {"red": 60, "red_tips": 3}), (31.0, {"red": 63}),
                         (35.5, {"red": 23, "red_tips": 1})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events],
                         [("hive_tip_auto", 20), ("unclassified", 3)])

    def test_teleop_rises_without_a_tip_are_foul_credits(self):
        rows = timeline([(5.0, {"clock": 119}), (20.0, {"blue": 5}), (40.0, {"blue": 25}), (60.0, {"blue": 32})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events],
                         [("minor_foul", 5), ("major_foul", 20), ("unclassified", 7)])
        self.assertEqual(summarize({"events": events})["blue"]["foul_credit"], 25)

    def test_tip_entered_with_its_score_is_not_also_a_foul(self):
        rows = timeline([(5.0, {"clock": 119}), (20.0, {"red": 20}), (21.0, {"red_tips": 1})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events], [("hive_tip_teleop", 20)])

    def test_two_tips_entered_together(self):
        rows = timeline([(5.0, {"clock": 119}), (20.0, {"red": 40, "red_tips": 2})])
        events = propose_events(rows)["events"]
        self.assertEqual([e["event_type"] for e in events], ["hive_tip_teleop", "hive_tip_teleop"])

    def test_lowered_tip_count_removes_the_latest_tip(self):
        rows = timeline([(5.0, {"clock": 119}), (20.0, {"red": 20, "red_tips": 1}),
                         (30.0, {"red": 0, "red_tips": 0})])
        proposal = propose_events(rows)
        # The TIP is withdrawn and its score drop cancels the rise: nothing is proposed.
        self.assertEqual(proposal["events"], [])
        self.assertEqual(proposal["corrections"][0]["removed_tip_t_s"], 20.0)
        self.assertEqual(proposal["corrections"][1]["cancelled_rise_t_s"], 20.0)

    def test_unmatched_drop_is_a_negative_adjustment(self):
        # The only rise was a TIP (explained, so not open to correction).
        rows = timeline([(5.0, {"clock": 110}), (20.0, {"red": 20, "red_tips": 1}), (25.0, {"red": 10})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events],
                         [("hive_tip_teleop", 20), ("unclassified", -10)])
        self.assertIn("correction", events[1]["note"])

    def test_partial_correction_shrinks_the_latest_rise(self):
        rows = timeline([(5.0, {"clock": 110}), (20.0, {"red": 30}), (25.0, {"red": 20})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events], [("unclassified", 20)])

    def test_missed_transition_still_reaches_teleop(self):
        rows = timeline([(5.0, {"clock": 119})])
        self.assertEqual([phase for _, phase in match_phases(rows)], [AUTO, TELEOP])

    def test_score_after_the_match_is_an_end_assessment(self):
        rows = timeline([(5.0, {"clock": 119}), (125.0, {"clock": 0}), (130.0, {"red": 18})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events], [("unclassified", 18)])
        self.assertIn("end-of-match", events[0]["note"])

    def test_previous_match_score_before_auto_is_ignored(self):
        # Athens Q11: the clip opens on the last match's score before AUTO starts.
        rows = [sample(t / 2, red=459, blue=59, red_tips=3, blue_tips=1) for t in range(6)]
        rows += [sample(3 + t / 2, red=0, blue=0, red_tips=0, blue_tips=0, clock=150 - t // 2) for t in range(6)]
        proposal = propose_events(rows)
        self.assertEqual(proposal["events"], [])
        self.assertEqual(proposal["phases"][0], (3.0, AUTO))

    def test_twenty_in_auto_without_a_tip_is_a_major_foul(self):
        rows = timeline([(10.0, {"blue": 20}), (20.0, {"blue": 28})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events],
                         [("major_foul", 20), ("unclassified", 8)])

    def test_clip_opening_on_the_previous_matchs_final_clock(self):
        rows = [sample(t / 2, red=88, blue=40, red_tips=4, blue_tips=2, clock=0) for t in range(6)]
        rows += timeline([(0.0, {"clock": 150}), (10.0, {"red": 20, "red_tips": 1, "clock": 140}),
                          (40.0, {"clock": 110})])
        for row in rows[6:]:
            row["t_s"] += 3.0
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [ENDED, AUTO, TELEOP])
        self.assertEqual(proposal["match_start_s"], 3.0)
        self.assertEqual([(e["event_type"], e["t_s"]) for e in proposal["events"]], [("hive_tip_auto", 13.0)])

    def test_clip_running_into_the_next_match_stops_at_its_auto(self):
        rows = timeline([(5.0, {"clock": 110}), (20.0, {"red": 20, "red_tips": 1}), (30.0, {"clock": 0}),
                         (35.0, {"clock": 150, "red": 0, "red_tips": 0}), (40.0, {"red": 20, "red_tips": 1})])
        proposal = propose_events(rows)
        self.assertEqual(proposal["match_end_s"], 35.0)
        self.assertEqual([(e["event_type"], e["t_s"]) for e in proposal["events"]], [("hive_tip_teleop", 20.0)])

    def test_tip_count_read_late_still_pairs_with_its_score(self):
        # Score readable at 20 s, the TIP counter only at 27 s (beyond the 3 s window).
        rows = timeline([(5.0, {"clock": 110}), (20.0, {"red": 20, "red_tips": None}),
                         (27.0, {"red_tips": 1}), (40.0, {"red": 25})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events],
                         [("hive_tip_teleop", 20), ("minor_foul", 5)])

    def test_partially_corrected_foul_is_no_longer_a_foul(self):
        rows = timeline([(5.0, {"clock": 110}), (20.0, {"red": 20}), (25.0, {"red": 15})])
        events = propose_events(rows)["events"]
        self.assertEqual([(e["event_type"], e["points"]) for e in events], [("unclassified", 15)])

    def test_cut_from_auto_to_the_final_clock_ends_the_match(self):
        rows = timeline([(10.0, {"clock": 135}), (20.0, {"clock": 0}), (25.0, {"red": 18})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [AUTO, ENDED])
        self.assertIn("end-of-match", proposal["events"][0]["note"])

    def test_clip_opening_in_the_countdown_keeps_its_tip_in_auto(self):
        rows = timeline([(0.0, {"clock": 6}), (2.0, {"red": 20, "red_tips": 1, "clock": 5}),
                         (8.0, {"clock": 120}), (20.0, {"clock": 108})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [TRANSITION, TELEOP])
        self.assertEqual(proposal["events"][0]["event_type"], "hive_tip_auto")

    def test_clip_opening_on_the_last_teleop_seconds_is_relabelled_teleop(self):
        rows = timeline([(0.0, {"clock": 6}), (2.0, {"red": 20, "red_tips": 1, "clock": 4}), (7.0, {"clock": 0})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [TELEOP, ENDED])
        self.assertEqual(proposal["events"][0]["event_type"], "hive_tip_teleop")

    def test_stream_cutting_into_the_next_match_at_two_fifteen_starts_auto(self):
        rows = timeline([(0.0, {"clock": 60}), (5.0, {"clock": 0}), (10.0, {"clock": 135}),
                         (12.0, {"red": 20, "red_tips": 1, "clock": 133})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [TELEOP, ENDED, AUTO])
        self.assertEqual(proposal["events"][-1]["event_type"], "hive_tip_auto")

    def test_clip_opening_on_the_last_teleop_seconds_then_the_next_match(self):
        rows = timeline([(0.0, {"clock": 3}), (5.0, {"clock": 150}), (10.0, {"clock": 145})])
        self.assertEqual([phase for _, phase in match_phases(rows)], [TELEOP, AUTO])

    def test_cut_from_two_minutes_to_the_final_clock_ends_the_match(self):
        # The countdown never shows 0:00, so 2:00 then 0:00 is a cut to the end.
        rows = timeline([(5.0, {"clock": 120}), (8.0, {"clock": 0}), (12.0, {"red": 18})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [AUTO, ENDED])
        self.assertIn("end-of-match", proposal["events"][0]["note"])

    def test_cut_from_the_countdown_to_the_final_clock_ends_the_match(self):
        rows = timeline([(5.0, {"clock": 121}), (6.0, {"clock": 7}), (8.0, {"clock": 0}), (12.0, {"red": 18})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [AUTO, TRANSITION, ENDED])
        self.assertIn("end-of-match", proposal["events"][0]["note"])

    def test_cut_from_the_end_into_the_next_matchs_teleop_ends_the_match_there(self):
        rows = timeline([(0.0, {"clock": 60}), (2.0, {"red": 20, "red_tips": 1}), (5.0, {"clock": 0}),
                         (10.0, {"clock": 100, "red": 5, "red_tips": 0}), (14.0, {"red": 25, "red_tips": 1})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [TELEOP, ENDED, TELEOP])
        self.assertEqual(proposal["match_end_s"], 10.0)
        self.assertEqual([(e["t_s"], e["event_type"]) for e in proposal["events"]], [(2.0, "hive_tip_teleop")])

    def test_clip_opening_on_the_last_final_score_then_cutting_into_teleop_reads_that_match(self):
        rows = timeline([(0.0, {"clock": 0, "red": 80, "red_tips": 3}), (5.0, {"clock": 100, "red": 0, "red_tips": 0}),
                         (8.0, {"red": 20, "red_tips": 1})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [ENDED, TELEOP])
        self.assertEqual(proposal["match_start_s"], 5.0)
        self.assertEqual([(e["t_s"], e["event_type"]) for e in proposal["events"]], [(8.0, "hive_tip_teleop")])

    def test_correction_cancels_the_latest_rises_that_add_up(self):
        # A 20 (major foul), then 5 and 15; a 20 drop withdraws the 5 and 15.
        rows = timeline([(0.0, {"clock": 100}), (2.0, {"red": 20}), (6.0, {"red": 25}), (10.0, {"red": 40}),
                         (14.0, {"red": 20})])
        proposal = propose_events(rows)
        self.assertEqual([(e["t_s"], e["event_type"], e["points"]) for e in proposal["events"]],
                         [(2.0, "major_foul", 20)])

    def test_stream_cutting_from_auto_into_the_next_matchs_auto_starts_a_new_match(self):
        rows = timeline([(0.0, {"clock": 140}), (2.0, {"red": 20, "red_tips": 1}),
                         (5.0, {"clock": 150, "red": 0, "red_tips": 0}), (8.0, {"blue": 20, "blue_tips": 1}),
                         (12.0, {"clock": 121}), (13.0, {"clock": 7}), (17.0, {"clock": 120}), (19.0, {"clock": 118})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [AUTO, AUTO, TRANSITION, TELEOP])
        self.assertEqual(proposal["match_start_s"], 5.0)
        self.assertEqual([(e["t_s"], e["scoring_alliance"]) for e in proposal["events"]], [(8.0, "blue")])

    def test_teleop_clock_jumping_up_is_a_cut_into_another_match(self):
        rows = timeline([(0.0, {"clock": 100}), (2.0, {"red": 20, "red_tips": 1}),
                         (5.0, {"clock": 115, "red": 0, "red_tips": 0}), (8.0, {"red": 20, "red_tips": 1})])
        proposal = propose_events(rows)
        self.assertEqual([phase for _, phase in proposal["phases"]], [TELEOP, TELEOP])
        self.assertEqual(proposal["match_end_s"], 5.0)
        self.assertEqual([e["t_s"] for e in proposal["events"]], [2.0])

    def test_no_readings(self):
        proposal = propose_events([sample(0), sample(0.5)])
        self.assertEqual(proposal["events"], [])
        self.assertIsNone(summarize(proposal)["red"]["last_stable_score"])


class FakeCapture:
    """A 10 fps video of ``frames`` frames whose decoding fails from ``fail_at`` on."""

    def __init__(self, frames, fail_at=None, raise_at=(), refuse_rewind=False, refuse_seek=False, decoded=None,
                 raise_seek=False):
        self.frames, self.fail_at, self.position, self.raise_seek = frames, fail_at, 0, raise_seek
        self.raise_at, self.refuse_rewind, self.refuse_seek = set(raise_at), refuse_rewind, refuse_seek
        # Positions of the frames decoded with read(), shared across reopened captures.
        self.decoded = decoded if decoded is not None else []

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 10.0, cv2.CAP_PROP_FRAME_COUNT: float(self.frames),
                cv2.CAP_PROP_POS_FRAMES: float(self.position)}[prop]

    def set(self, prop, value):
        if self.raise_seek:
            raise cv2.error("seek not supported")
        if not (self.refuse_seek or (self.refuse_rewind and value == 0)):
            self.position = int(value)

    def _ok(self):
        position = self.position
        self.position += 1
        if position in self.raise_at:
            raise cv2.error("corrupt frame")
        return self.fail_at is None or position < self.fail_at

    def grab(self):
        return self._ok()

    def read(self):
        self.decoded.append(self.position)
        return (True, np.zeros((300, 1920, 3), np.uint8)) if self._ok() else (False, None)

    def release(self):
        pass


class FakeEngine:
    def __init__(self, fail_reads=(), fail_locate=0, text="0"):
        self.reads, self.fail_reads, self.fail_locate, self.text = 0, set(fail_reads), fail_locate, text

    def text_rec(self, crops):
        self.reads += 1
        if self.reads in self.fail_reads:
            raise RuntimeError("ONNX error")
        return [(self.text, 0.9) for _ in crops], 0.0

    def __call__(self, frame):
        # Full-frame OCR used to locate the overlay: fail the first probes.
        if self.fail_locate:
            self.fail_locate -= 1
            raise RuntimeError("bad probe frame")
        return [[[[719, 111], [778, 111], [778, 141], [719, 141]], "Red", 0.99],
                [[[1141, 108], [1207, 108], [1207, 143], [1141, 143]], "Blue", 0.99]], 0.0


class ReadOverlayTests(unittest.TestCase):
    LAYOUT = layout_from_labels((719, 111, 778, 141), (1141, 108, 1207, 143), (300, 1920, 3))

    def read(self, capture, engine=None, layout=LAYOUT):
        from pathlib import Path
        from unittest.mock import patch
        from control_vision.pipeline import live_score
        with patch.object(live_score, "_engine", return_value=engine or FakeEngine()), \
                patch.object(live_score.cv2, "VideoCapture", return_value=capture):
            return live_score.read_live_overlay(Path("match.mp4"), layout=layout, sample_fps=2.0)

    def test_one_ocr_failure_skips_only_that_frame(self):
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = self.read(FakeCapture(100), FakeEngine(fail_reads={3}))
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["samples"]), 19)

    def test_repeated_ocr_failures_stop_the_read_and_say_so(self):
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = self.read(FakeCapture(100), FakeEngine(fail_reads=set(range(4, 30))))
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["samples"]), 3)

    def test_a_bad_probe_frame_moves_on_to_the_next_probe(self):
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = self.read(FakeCapture(100), FakeEngine(fail_locate=2), layout=None)
        self.assertEqual(result["layout"]["red_score"], list(self.LAYOUT.red_score))

    def test_a_decoder_exception_skips_only_that_frame(self):
        # Frame 10 is read (sampled every 5th frame), frame 11 is only grabbed.
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = self.read(FakeCapture(100, raise_at={10}))
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["samples"]), 19)
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = self.read(FakeCapture(100, raise_at={11}))
        self.assertFalse(result["complete"])
        self.assertIn("1.1", result["stopped_early"])
        self.assertEqual(len(result["samples"]), 3)

    def test_a_probe_decoder_exception_moves_on_to_the_next_probe(self):
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            # Frame 50 is the first probe (and later a sampled frame, skipped).
            result = self.read(FakeCapture(100, raise_at={50}), layout=None)
        self.assertEqual(result["layout"]["red_score"], list(self.LAYOUT.red_score))
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["samples"]), 19)

    def test_a_refused_rewind_reopens_the_video_so_times_start_at_zero(self):
        from pathlib import Path
        from unittest.mock import patch
        from control_vision.pipeline import live_score
        stuck, fresh = FakeCapture(100, refuse_rewind=True), FakeCapture(100)
        with patch.object(live_score, "_engine", return_value=FakeEngine()), \
                patch.object(live_score.cv2, "VideoCapture", side_effect=[stuck, fresh]), \
                self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = live_score.read_live_overlay(Path("match.mp4"), sample_fps=2.0)
        self.assertEqual(result["samples"][0]["t_s"], 0.0)
        self.assertEqual(len(result["samples"]), 20)
        self.assertEqual(fresh.position, 100)

    def test_refused_probe_seeks_still_probe_the_right_frames(self):
        from pathlib import Path
        from unittest.mock import patch
        from control_vision.pipeline import live_score
        decoded: list = []
        engine = FakeEngine(fail_locate=1)  # the first probe (frame 50) is searched but fails
        with patch.object(live_score, "_engine", return_value=engine),                 patch.object(live_score.cv2, "VideoCapture",
                             side_effect=lambda *_: FakeCapture(100, refuse_seek=True, decoded=decoded)),                 self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = live_score.read_live_overlay(Path("match.mp4"), sample_fps=2.0)
        self.assertEqual(decoded[:2], [50, 35])
        self.assertEqual(result["samples"][0]["t_s"], 0.0)
        self.assertEqual(len(result["samples"]), 20)
        self.assertTrue(result["complete"])

    def test_a_seek_that_raises_reopens_instead_of_escaping(self):
        from pathlib import Path
        from unittest.mock import patch
        from control_vision.pipeline import live_score
        decoded: list = []
        with patch.object(live_score, "_engine", return_value=FakeEngine()),                 patch.object(live_score.cv2, "VideoCapture",
                             side_effect=lambda *_: FakeCapture(100, raise_seek=True, decoded=decoded)),                 self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = live_score.read_live_overlay(Path("match.mp4"), sample_fps=2.0)
        self.assertEqual(decoded[0], 50)
        self.assertEqual(result["samples"][0]["t_s"], 0.0)
        self.assertTrue(result["complete"])

    def test_no_searchable_probe_is_a_reader_failure_not_a_missing_overlay(self):
        for capture, engine in ((FakeCapture(100), FakeEngine(fail_locate=5)),
                                (FakeCapture(100, raise_at={50, 35, 65, 20, 80}), FakeEngine())):
            with self.subTest(engine=engine.fail_locate),                     self.assertLogs("control_vision.pipeline.live_score", level="WARNING"),                     self.assertRaisesRegex(RuntimeError, "probe frames .* may be broken"):
                self.read(capture, engine, layout=None)

    def test_an_unreadable_last_sample_marks_the_read_incomplete(self):
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = self.read(FakeCapture(100), FakeEngine(fail_reads={20}))
        self.assertFalse(result["complete"])
        self.assertIn("from 9.5 s", result["stopped_early"])
        self.assertEqual(result["read_until_s"], 9.0)

    def test_an_overlay_with_nothing_readable_is_an_error_not_a_scoreless_read(self):
        with self.assertRaisesRegex(RuntimeError, "No score, clock or HIVE TIP count"):
            self.read(FakeCapture(100), FakeEngine(text="~~"))

    def test_ocr_failing_from_the_start_says_why_no_frame_was_read(self):
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"),                 self.assertRaisesRegex(RuntimeError, "Could not read any frame .*: 5 sampled frames in a row"):
            self.read(FakeCapture(100), FakeEngine(fail_reads=set(range(1, 30))))

    def test_complete_read(self):
        result = self.read(FakeCapture(100))
        self.assertTrue(result["complete"])
        self.assertIsNone(result["stopped_early"])
        self.assertEqual(len(result["samples"]), 20)

    def test_a_decode_failure_partway_is_reported(self):
        with self.assertLogs("control_vision.pipeline.live_score", level="WARNING"):
            result = self.read(FakeCapture(100, fail_at=42))
        self.assertFalse(result["complete"])
        self.assertIn("4.2", result["stopped_early"])
        self.assertLess(result["read_until_s"], 4.2)
        self.assertEqual(result["duration_s"], 10.0)


class EvalCacheTests(unittest.TestCase):
    """scripts/eval_live_score.py: a truncated result file is redone, not fatal."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "scripts" / "eval_live_score.py"
        spec = importlib.util.spec_from_file_location("eval_live_score", path)
        cls.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.script)

    def _footage(self, folder):
        footage = folder / "footage"
        footage.mkdir()
        (footage / "FOOTAGE_MANIFEST.csv").write_text(
            "file,event_code,match_label_on_screen\nclip.mp4,TEST,Qualification 3\n", encoding="utf-8")
        return footage

    def test_read_redoes_a_truncated_result(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            footage, out = self._footage(folder), folder / "results"
            out.mkdir()
            (out / "clip.json").write_text('{"file": "clip.mp4", "samp', encoding="utf-8")
            reading = {"layout": {}, "samples": [], "complete": True, "stopped_early": None}
            with patch.object(self.script, "read_live_overlay", return_value=reading) as reader:
                self.script.read(footage, folder / "official", out, [])
            reader.assert_called_once()
            import json
            result = json.loads((out / "clip.json").read_text(encoding="utf-8"))
            self.assertEqual(result["match"], 3)
            self.assertTrue(result["complete"])
            self.assertEqual(list(out.glob("*.tmp")), [])

    def test_report_lists_a_truncated_result_instead_of_failing(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "clip.json").write_text('{"file": "clip.mp4", "samp', encoding="utf-8")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                self.script.report(out)
            self.assertIn("Alliance results compared: 0", printed.getvalue())
            self.assertIn("clip: result file is unreadable", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
