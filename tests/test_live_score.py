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
        texts = iter([("43", 0.99), ("0", 0.98), ("", 0.0), ("", 0.0), ("1:36", 0.97)])
        row = read_frame(lambda crops: ([next(texts) for _ in crops], 0.0), frame, layout)
        self.assertEqual((row["red_score"], row["blue_score"], row["red_tips"], row["clock_s"]), (43, 0, None, 96))


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

    def test_no_readings(self):
        proposal = propose_events([sample(0), sample(0.5)])
        self.assertEqual(proposal["events"], [])
        self.assertIsNone(summarize(proposal)["red"]["last_stable_score"])


if __name__ == "__main__":
    unittest.main()
