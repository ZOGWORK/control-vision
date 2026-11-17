import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from control_vision.pipeline.identity_recovery import IdentityRecovery, RecoveryRules
from control_vision.pipeline.tracking import _overlap_of_smaller, _tracker_config, track_video

# 120 px square mapped to the 12 ft field: 10 px per foot.
CORNERS = [[0, 0], [120, 0], [120, 120], [0, 120]]
FPS = 8.0


def _px(feet: float) -> float:
    return feet * 10.0


class _Capture:
    def __init__(self, frames: int):
        self.frames = frames
        self.reads = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return FPS if prop == cv2.CAP_PROP_FPS else self.frames

    def read(self):
        self.reads += 1
        if self.reads > self.frames:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        pass


BGR = {"red": (0, 0, 255), "blue": (255, 0, 0)}


def _box(track_id: int, x_ft: float, y_ft: float, conf: float = 0.9):
    x, y = _px(x_ft), _px(y_ft)
    return SimpleNamespace(cls=[0], xyxy=np.array([[x - 4, y - 8, x + 4, y]]), conf=[conf], id=[track_id])


def _paint(frame, track_id, x_ft, y_ft, colour):
    """Give the box a quarter of alliance colour along its bottom, like a bumper."""
    if colour:
        x, y = int(_px(x_ft)), int(_px(y_ft))
        frame[y - 2:y, x - 4:x + 4] = BGR[colour]


class _Logs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def run_script(script, seeds, confidence=0.25, track_kwargs=None, paint=None, alliances="red", reseeds=None):
    """Run track_video on a scripted tracker output.

    ``script`` is a list, one entry per frame at 8 fps, of (track id, x ft, y ft)
    detections. Seeds are (robot id, x ft, y ft). ``paint`` maps track IDs to the
    alliance colour drawn on their box (red unless given; None draws nothing).
    ``alliances`` is the alliance of every seeded robot, or a dict, or None.
    Returns rows, log messages, and the keyword arguments of every model.track call.
    """
    calls = []

    def track(frame, **kwargs):
        calls.append(kwargs)
        if track_kwargs is not None:
            track_kwargs(kwargs)
        detections = script[len(calls) - 1]
        for item in detections:
            _paint(frame, *item, (paint or {}).get(item[0], "red"))
        return [SimpleNamespace(boxes=[_box(*item) for item in detections])]

    logs = _Logs()
    logger = logging.getLogger(f"track-recovery-test-{id(logs)}")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logs)
    seed_points = [(rid, _px(x), _px(y)) for rid, x, y in seeds]
    if isinstance(alliances, str):
        alliances = {rid: alliances for rid, _, _ in seeds}
    with tempfile.TemporaryDirectory() as folder:
        model_path = Path(folder) / "robot.onnx"
        model_path.touch()
        with patch("control_vision.pipeline.tracking.cv2.VideoCapture", return_value=_Capture(len(script))), \
             patch("ultralytics.YOLO", return_value=SimpleNamespace(track=track)):
            rows, _, _ = track_video(Path("match.mp4"), model_path, CORNERS, seed_points, FPS,
                                     lambda: False, lambda *_: None, logger,
                                     confidence_threshold=confidence, alliances=alliances,
                                     reseeds=[(rid, t, _px(x), _px(y)) for rid, t, x, y in reseeds or []])
    return rows, logs.messages, calls


def robot_times(rows, robot_id):
    return [round(row[2], 3) for row in rows if row[0] == robot_id]


def recovered(messages):
    return [message for message in messages if message.startswith("Recovered robot")]


class TrackerThresholdTests(unittest.TestCase):
    def test_user_confidence_reaches_the_tracker_config_file(self):
        import yaml
        seen = []

        def read_config(kwargs):
            seen.append(yaml.safe_load(Path(kwargs["tracker"]).read_text(encoding="utf-8")))

        rows, _, calls = run_script([[(1, 3, 3)]] * 3, [(1, 3, 3)], confidence=0.2,
                                    track_kwargs=read_config)
        self.assertEqual(len(rows), 3)
        self.assertEqual(calls[0]["conf"], 0.2)
        config = seen[0]
        self.assertEqual(config["track_high_thresh"], 0.2)
        self.assertEqual(config["new_track_thresh"], 0.2)
        self.assertLess(config["track_low_thresh"], 0.2)
        self.assertEqual(config["tracker_type"], "botsort")
        # The run's file is removed afterwards and the packaged config is untouched.
        self.assertFalse(Path(calls[0]["tracker"]).exists())

    def test_low_confidence_keeps_low_threshold_below_it(self):
        config = _tracker_config({"track_low_thresh": 0.1, "gmc_method": "sparseOptFlow"}, 0.08)
        self.assertEqual(config["track_high_thresh"], 0.08)
        self.assertEqual(config["new_track_thresh"], 0.08)
        self.assertAlmostEqual(config["track_low_thresh"], 0.04)
        self.assertEqual(config["gmc_method"], "sparseOptFlow")

    def test_overrides_and_invalid_confidence(self):
        with tempfile.TemporaryDirectory() as folder:
            reid = Path(folder) / "reid.onnx"
            config = _tracker_config({"track_low_thresh": 0.1, "with_reid": False}, 0.4, "none", reid)
        self.assertEqual(config["gmc_method"], "none")
        self.assertTrue(config["with_reid"])
        self.assertEqual(config["model"], str(reid.resolve()))
        self.assertEqual(config["track_low_thresh"], 0.1)
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                _tracker_config({}, bad)

    def test_packaged_config_keeps_motion_compensation_by_default(self):
        import yaml
        from control_vision.pipeline.tracking import TRACKER_CONFIG
        config = yaml.safe_load(TRACKER_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["gmc_method"], "sparseOptFlow")
        self.assertEqual(_tracker_config(config, 0.25)["gmc_method"], "sparseOptFlow")
        self.assertEqual(_tracker_config(config, 0.25, "none")["gmc_method"], "none")


def _lost_then(new_tracks, gap_frames=1, tail=6, lost_frame=8):
    """Robot 1 on track 1 at (3, 3) ft for ``lost_frame`` frames, then a gap,
    then ``new_tracks`` (track id, x, y) for ``tail`` frames."""
    script = [[(1, 3, 3)] for _ in range(lost_frame)]
    script += [[] for _ in range(gap_frames)]
    script += [list(new_tracks) for _ in range(tail)]
    return script


class IdentityRecoveryRunTests(unittest.TestCase):
    def test_one_clean_candidate_is_recovered_with_its_earlier_positions(self):
        # Last seen at 0.875 s; track 5 is born at 1.125 s one foot away.
        rows, messages, _ = run_script(_lost_then([(5, 4, 3)]), [(1, 3, 3)])
        self.assertEqual(len(recovered(messages)), 1, messages)
        self.assertIn("track 1 -> new track 5", recovered(messages)[0])
        self.assertIn("1.00 ft from last position", recovered(messages)[0])
        times = robot_times(rows, 1)
        # Recovery happens once the robot has been missing 0.5 s (at 1.375 s);
        # the new track's positions from its birth at 1.125 s are kept.
        self.assertIn(1.125, times)
        self.assertIn(1.625, times)
        self.assertEqual(len(times), len(set(times)))
        self.assertEqual(len(times), 8 + 6)

    def test_two_candidates_in_the_gate_block_recovery(self):
        rows, messages, _ = run_script(_lost_then([(5, 4, 3), (6, 3, 4)]), [(1, 3, 3)])
        self.assertEqual(recovered(messages), [])
        self.assertEqual(max(robot_times(rows, 1)), 0.875)
        self.assertTrue(any("another unassigned track is inside the gate" in m for m in messages), messages)

    def test_two_lost_robots_cannot_share_one_new_track(self):
        script = [[(1, 3, 3), (2, 5, 3)] for _ in range(8)] + [[]] + [[(5, 4, 3)] for _ in range(6)]
        rows, messages, _ = run_script(script, [(1, 3, 3), (2, 5, 3)])
        self.assertEqual(recovered(messages), [])
        self.assertEqual(max(robot_times(rows, 1)), 0.875)
        self.assertEqual(max(robot_times(rows, 2)), 0.875)
        self.assertTrue(any("more than one missing robot" in m for m in messages), messages)

    def test_new_track_beyond_the_speed_gate_is_not_recovered(self):
        # Born 0.125 s after the last sighting: gate is 15 * 0.125 + 1 = 2.875 ft.
        rows, messages, _ = run_script(_lost_then([(5, 6.2, 3)], gap_frames=0, tail=20), [(1, 3, 3)])
        self.assertEqual(recovered(messages), [])
        self.assertEqual(max(robot_times(rows, 1)), 0.875)

    def test_new_track_just_inside_the_speed_gate_is_recovered(self):
        rows, messages, _ = run_script(_lost_then([(5, 5.8, 3)], gap_frames=0), [(1, 3, 3)])
        self.assertEqual(len(recovered(messages)), 1, messages)

    def test_no_recovery_after_the_maximum_absence(self):
        # 10.5 s gap at 8 fps.
        rows, messages, _ = run_script(_lost_then([(5, 3, 3)], gap_frames=84), [(1, 3, 3)])
        self.assertEqual(recovered(messages), [])
        self.assertEqual(max(robot_times(rows, 1)), 0.875)
        self.assertTrue(any("identity recovery stopped" in m for m in messages), messages)

    def test_recovery_just_before_the_maximum_absence(self):
        # Born 9.6 s after the last sighting, alone on the field.
        rows, messages, _ = run_script(_lost_then([(5, 3, 3)], gap_frames=76), [(1, 3, 3)])
        self.assertEqual(len(recovered(messages)), 1, messages)

    def test_tracked_robot_is_never_reassigned(self):
        # Robot 1 stays on track 1 while a second box (track 5) appears next to it.
        script = [[(1, 3, 3)] for _ in range(8)] + [[(1, 3, 3), (5, 3.5, 3)] for _ in range(10)]
        rows, messages, _ = run_script(script, [(1, 3, 3)])
        self.assertEqual(recovered(messages), [])
        positions = {(round(row[3], 2), round(row[4], 2)) for row in rows if row[0] == 1}
        self.assertEqual(positions, {(3.0, 3.0)})

    def test_tracked_robot_is_not_taken_by_a_lost_robot_either(self):
        # Robot 2 is lost; robot 1's own track stays visible and is never handed over.
        script = [[(1, 3, 3), (2, 6, 6)] for _ in range(8)] + [[(1, 3, 3)] for _ in range(10)]
        rows, messages, _ = run_script(script, [(1, 3, 3), (2, 6, 6)])
        self.assertEqual(recovered(messages), [])
        self.assertEqual(max(robot_times(rows, 2)), 0.875)
        self.assertEqual(len(robot_times(rows, 1)), 18)

    def test_unseeded_late_entrant_is_never_assigned(self):
        # The seed never matches in the opening window; a robot appears later on the seed.
        script = [[(1, 10, 10)] for _ in range(30)] + [[(1, 10, 10), (7, 2, 2)] for _ in range(10)]
        with self.assertRaises(RuntimeError):
            run_script(script, [(1, 2, 2)])

    def test_old_track_returning_after_recovery_is_not_used_again(self):
        script = _lost_then([(5, 4, 3)]) + [[(5, 4, 3), (1, 8, 8)] for _ in range(5)] + [[(1, 8, 8)] for _ in range(6)]
        rows, messages, _ = run_script(script, [(1, 3, 3)])
        self.assertEqual(len(recovered(messages)), 1)
        positions = {(round(row[3], 1), round(row[4], 1)) for row in rows if row[0] == 1}
        self.assertNotIn((8.0, 8.0), positions)


def reseeded(messages):
    return [message for message in messages if message.startswith("Re-seeded robot")]


class SeedWindowReseedTests(unittest.TestCase):
    """A seed that lands on a short-lived duplicate box moves to the robot's real track."""

    def test_seed_on_a_vanishing_duplicate_moves_to_the_overlapping_track(self):
        # Frame 0: the robot (track 1) and a duplicate box 0.3 ft to its right
        # (track 2), which the click is nearest to. Track 2 is never seen again.
        script = [[(1, 3, 3), (2, 3.3, 3)]] + [[(1, 3, 3)] for _ in range(19)]
        rows, messages, _ = run_script(script, [(1, 3.3, 3)])
        self.assertIn("Seeded robot 1 to detector track 2", "\n".join(messages))
        self.assertEqual(reseeded(messages), ["Re-seeded robot 1 from track 2, which vanished within the seed "
                                              "window, to overlapping track 1"])
        self.assertEqual(len(robot_times(rows, 1)), 20)

    def test_two_overlapping_boxes_leave_the_seed_unchanged(self):
        # Tracks 1 and 3 both overlap the vanished duplicate: no single answer.
        script = [[(1, 3, 3), (2, 3.3, 3), (3, 3.6, 3)]] + [[(1, 3, 3), (3, 3.6, 3)] for _ in range(19)]
        rows, messages, _ = run_script(script, [(1, 3.3, 3)])
        self.assertEqual(reseeded(messages), [])
        self.assertEqual(robot_times(rows, 1), [0.0])

    def test_box_that_does_not_overlap_is_not_used(self):
        script = [[(1, 5, 3), (2, 3, 3)]] + [[(1, 5, 3)] for _ in range(19)]
        rows, messages, _ = run_script(script, [(1, 3, 3)])
        self.assertEqual(reseeded(messages), [])
        self.assertEqual(robot_times(rows, 1), [0.0])

    def test_no_reseed_after_the_seed_window(self):
        # The seeded track lasts 3.5 s, then vanishes next to an older overlapping track.
        script = [[(1, 3, 3), (2, 3.3, 3)] for _ in range(28)] + [[(1, 3, 3)] for _ in range(10)]
        rows, messages, _ = run_script(script, [(1, 3.3, 3)])
        self.assertEqual(reseeded(messages), [])
        self.assertEqual(max(robot_times(rows, 1)), 3.375)

    def test_overlap_of_smaller(self):
        self.assertEqual(_overlap_of_smaller((0, 0, 10, 10), (2, 2, 6, 6)), 1.0)
        self.assertEqual(_overlap_of_smaller((0, 0, 10, 10), (5, 0, 15, 10)), 0.5)
        self.assertEqual(_overlap_of_smaller((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)


class UserReseedTests(unittest.TestCase):
    """Mid-match re-seeds: (robot id, time s, x ft, y ft) in these tests."""

    def test_reseed_brings_back_a_robot_after_its_recovery_window_closed(self):
        # Lost at 0.875 s; track 5 appears 10.5 s later, too late for automatic recovery.
        script = _lost_then([(5, 3, 3)], gap_frames=84, tail=10)
        rows, messages, _ = run_script(script, [(1, 3, 3)], reseeds=[(1, 11.5, 3, 3)])
        self.assertTrue(any("identity recovery stopped" in m for m in messages), messages)
        self.assertIn("Re-seeded robot 1 at 11.50s by the user: track 1 -> 5; 0 earlier positions kept",
                      messages)
        times = robot_times(rows, 1)
        self.assertEqual([t for t in times if t > 1], [11.5 + 0.125 * i for i in range(10)])

    def test_reseed_keeps_the_new_tracks_positions_since_the_robot_vanished(self):
        # Two candidates block automatic recovery; the user picks track 5 at 1.625 s.
        script = _lost_then([(5, 4, 3), (6, 3, 4)])
        rows, messages, _ = run_script(script, [(1, 3, 3)], reseeds=[(1, 1.625, 4, 3)])
        self.assertEqual(recovered(messages), [])
        self.assertIn("Re-seeded robot 1 at 1.62s by the user: track 1 -> 5; 4 earlier positions kept", messages)
        # Positions are smoothed over +-2 samples, so check past the switch.
        positions = {(round(row[3], 1), round(row[4], 1)) for row in rows if row[0] == 1 and row[2] >= 1.375}
        self.assertEqual(positions, {(4.0, 3.0)})
        times = robot_times(rows, 1)
        self.assertEqual(len(times), len(set(times)))
        self.assertEqual(min(t for t in times if t > 1), 1.125)

    def test_reseed_onto_another_robots_track_releases_that_robot(self):
        script = [[(1, 3, 3), (2, 8, 8)] for _ in range(16)]
        rows, messages, _ = run_script(script, [(1, 3, 3), (2, 8, 8)], reseeds=[(1, 1.0, 8, 8)])
        self.assertTrue(any(m.startswith("Robot 2 released from track 2 at 1.00s") for m in messages), messages)
        self.assertEqual(max(robot_times(rows, 2)), 0.875)
        after = {(round(row[3], 1), round(row[4], 1)) for row in rows if row[0] == 1 and row[2] >= 1.25}
        self.assertEqual(after, {(8.0, 8.0)})

    def test_click_on_no_robot_is_not_applied(self):
        script = [[(1, 3, 3)] for _ in range(20)]
        rows, messages, _ = run_script(script, [(1, 3, 3)], reseeds=[(1, 1.0, 9, 9)])
        self.assertIn("Re-seed of robot 1 at 1.00s was not applied: no tracked robot box at the click within 1.0s",
                      messages)
        self.assertEqual(len(robot_times(rows, 1)), 20)

    def test_click_on_the_robots_own_track_changes_nothing(self):
        script = [[(1, 3, 3)] for _ in range(20)]
        rows, messages, _ = run_script(script, [(1, 3, 3)], reseeds=[(1, 1.0, 3, 3)])
        self.assertIn("Re-seed of robot 1 at 1.00s confirms its current track 1", messages)
        self.assertEqual(len(robot_times(rows, 1)), 20)

    def test_a_click_that_finds_no_box_does_not_hold_up_another(self):
        # After the seed window, robot 2 is lost at 4.0 s with two candidates (no automatic recovery). At
        # 4.0 s the user clicks empty field for robot 1 and track 5 for robot 2; robot 2's must apply at once.
        script = ([[(1, 3, 3), (2, 8, 8)] for _ in range(32)]
                  + [[(1, 3, 3), (5, 8, 8), (6, 6.5, 8)] for _ in range(16)])
        rows, messages, _ = run_script(script, [(1, 3, 3), (2, 8, 8)],
                                       reseeds=[(1, 4.0, 10, 1), (2, 4.0, 8, 8)])
        self.assertTrue(any(m.startswith("Re-seeded robot 2 at 4.00s by the user") for m in messages), messages)
        self.assertTrue(any(m.startswith("Re-seed of robot 1 at 4.00s was not applied") for m in messages), messages)

    def test_a_click_is_not_applied_after_its_window(self):
        # Nothing is at the click until 2.125 s, 1.125 s after it: too late, even though a box is there then.
        script = [[(1, 3, 3)] for _ in range(17)] + [[(1, 3, 3), (7, 9, 9)] for _ in range(6)]
        rows, messages, _ = run_script(script, [(1, 3, 3)], reseeds=[(1, 1.0, 9, 9)])
        self.assertEqual([m for m in messages if m.startswith("Re-seeded")], [])
        self.assertIn("Re-seed of robot 1 at 1.00s was not applied: no tracked robot box at the click within 1.0s",
                      messages)

    def test_reseed_onto_a_track_another_robot_was_moved_off_keeps_its_positions(self):
        # 1.0 s: robot 1 is re-seeded onto robot 2's track 2, so track 1 is unowned.
        # 2.0 s: robot 2 is re-seeded onto track 1 and keeps its positions from 1.125 s to 1.875 s.
        script = [[(1, 3, 3), (2, 8, 8)] for _ in range(24)]
        rows, messages, _ = run_script(script, [(1, 3, 3), (2, 8, 8)],
                                       reseeds=[(1, 1.0, 8, 8), (2, 2.0, 3, 3)])
        self.assertIn("Re-seeded robot 2 at 2.00s by the user: track none -> 1; 7 earlier positions kept", messages)
        times = robot_times(rows, 2)
        self.assertEqual([t for t in times if t > 1], [1.125 + 0.125 * i for i in range(15)])

    def test_a_reseed_after_the_last_frame_is_reported(self):
        script = [[(1, 3, 3)] for _ in range(8)]
        rows, messages, _ = run_script(script, [(1, 3, 3)], reseeds=[(1, 50.0, 3, 3)])
        self.assertIn("Re-seed of robot 1 at 50.00s was not applied: the video ended before a tracked robot box "
                      "was found at the click", messages)

    def test_invalid_reseeds_are_rejected_before_processing(self):
        script = [[(1, 3, 3)] for _ in range(4)]
        for bad in ([(2, 1.0, 3, 3)], [(1, -1.0, 3, 3)], [(1, float("nan"), 3, 3)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_script(script, [(1, 3, 3)], reseeds=bad)


class AllianceColourRunTests(unittest.TestCase):
    def test_other_alliance_colour_blocks_recovery(self):
        rows, messages, _ = run_script(_lost_then([(5, 4, 3)]), [(1, 3, 3)], paint={5: "blue"})
        self.assertEqual(recovered(messages), [])
        self.assertTrue(any("alliance colour does not match" in m for m in messages), messages)

    def test_no_alliance_colour_blocks_recovery(self):
        rows, messages, _ = run_script(_lost_then([(5, 4, 3)]), [(1, 3, 3)], paint={5: None})
        self.assertEqual(recovered(messages), [])

    def test_blue_robot_is_recovered_on_a_blue_track(self):
        rows, messages, _ = run_script(_lost_then([(5, 4, 3)]), [(1, 3, 3)],
                                       paint={1: "blue", 5: "blue"}, alliances="blue")
        self.assertEqual(len(recovered(messages)), 1, messages)

    def test_without_alliances_lost_robots_stay_lost(self):
        rows, messages, _ = run_script(_lost_then([(5, 4, 3)]), [(1, 3, 3)], alliances=None)
        self.assertEqual(recovered(messages), [])
        self.assertTrue(any("not re-linked to new tracks" in m for m in messages), messages)

    def test_worker_passes_alliances_to_yolo_tracking(self):
        from control_vision.app.main_window import ProcessingWorker
        logger = logging.getLogger("worker-alliance-test")
        worker = ProcessingWorker((Path("match.mp4"), Path("robot.onnx"), [], [(1, 20, 30)], 8.0, logger,
                                   None, 0.2, None, "", "yolo", {1: "blue"}))
        with patch("control_vision.app.main_window.track_video", autospec=True,
                   return_value=([], [], [])) as tracker:
            worker.run()
        self.assertEqual(tracker.call_args.kwargs["alliances"], {1: "blue"})
        self.assertEqual(tracker.call_args.kwargs["confidence_threshold"], 0.2)


class AllianceColourRuleTests(unittest.TestCase):
    def state(self, reference=0.1, alliance="red"):
        state = IdentityRecovery(alliances={1: alliance})
        colour = (reference, 0.0) if alliance == "red" else (0.0, reference)
        for index in range(8):
            state.observe_robot(1, index / 8, 3.0, 3.0, colour)
        return state

    def test_reference_is_the_median_of_the_first_sightings_only(self):
        state = self.state(0.1)
        for index in range(9):
            state.observe_robot(1, 2.0 + index, 3.0, 3.0, (0.9, 0.0))
        self.assertAlmostEqual(state.colour_reference(1), 0.1)
        self.assertFalse(state.wants_colour(1))

    def test_colour_bounds(self):
        state = self.state(0.1)
        self.assertTrue(state.colour_matches(1, (0.1, 0.0)))
        self.assertTrue(state.colour_matches(1, (0.29, 0.02)))
        self.assertTrue(state.colour_matches(1, (0.034, 0.0)))
        self.assertFalse(state.colour_matches(1, (0.31, 0.0)))   # far more red: a different robot
        self.assertFalse(state.colour_matches(1, (0.03, 0.0)))   # far less red
        self.assertFalse(state.colour_matches(1, (0.1, 0.021)))  # other alliance colour present
        self.assertFalse(state.colour_matches(1, (0.015, 0.016)))  # other colour exceeds own
        self.assertFalse(state.colour_matches(1, None))

    def test_reference_is_frozen_after_a_relink(self):
        from control_vision.pipeline.identity_recovery import Recovery
        state = IdentityRecovery(alliances={1: "red"})
        state.observe_robot(1, 0.0, 3.0, 3.0, (0.1, 0.0))
        state.apply(Recovery(1, 1, 5, 1.0, 0.6, 0.5, 10.0))
        self.assertFalse(state.wants_colour(1))
        state.observe_robot(1, 1.1, 3.0, 3.0, (0.5, 0.0))
        self.assertAlmostEqual(state.colour_reference(1), 0.1)
        self.assertIn(5, state.ever_assigned)

    def test_low_reference_still_needs_some_own_colour(self):
        state = self.state(0.0)
        self.assertFalse(state.colour_matches(1, (0.0, 0.0)))
        self.assertTrue(state.colour_matches(1, (0.04, 0.0)))
        self.assertFalse(state.colour_matches(1, (0.06, 0.0)))
        # Within the bounds, but more of the other alliance's colour than its own.
        self.assertFalse(state.colour_matches(1, (0.012, 0.018)))

    def test_blue_alliance_uses_blue_as_own_colour(self):
        state = self.state(0.1, "blue")
        self.assertTrue(state.colour_matches(1, (0.0, 0.1)))
        self.assertFalse(state.colour_matches(1, (0.1, 0.0)))

    def test_robot_without_alliance_or_reference_never_matches(self):
        self.assertFalse(IdentityRecovery(alliances={}).colour_matches(1, (0.1, 0.0)))
        self.assertFalse(IdentityRecovery(alliances={1: "red"}).colour_matches(1, (0.1, 0.0)))
        self.assertFalse(IdentityRecovery(alliances={1: "green"}).colour_matches(1, (0.1, 0.0)))


class IdentityRecoveryRuleTests(unittest.TestCase):
    def setUp(self):
        self.state = IdentityRecovery(RecoveryRules())
        self.state.note_assignment(1)
        self.state.observe_robot(1, 1.0, 3.0, 3.0)
        self.state.observe_tracks(0.0, {1: (3.0, 3.0)})

    def recover(self, t_s, visible, mapping=None):
        self.state.observe_tracks(t_s, visible)
        return self.state.recover(t_s, visible, mapping or {1: 1})

    def test_track_that_existed_before_the_robot_vanished_is_not_a_candidate(self):
        # Track 9 was born next to the robot while the robot was still tracked.
        self.state.observe_tracks(0.5, {9: (3.2, 3.0)})
        self.assertEqual(self.recover(2.0, {9: (3.5, 3.0)}), [])

    def test_older_unassigned_track_in_the_gate_blocks_a_clean_candidate(self):
        self.state.observe_tracks(0.5, {9: (9.0, 9.0)})
        self.assertEqual(self.recover(1.25, {5: (3.5, 3.0), 9: (9.0, 9.0)}), [])
        # At 2.0 s the gate is 16 ft and covers track 9 as well.
        self.assertEqual(self.recover(2.0, {5: (3.5, 3.0), 9: (9.0, 9.0)}), [])

    def test_rival_seen_earlier_in_the_absence_still_blocks_after_it_disappears(self):
        self.assertEqual(self.recover(1.25, {9: (4.0, 3.0)}), [])
        # Track 9 is gone; track 5 is alone now but 9 could have been the robot.
        self.assertEqual(self.recover(1.5, {5: (3.5, 3.0)}), [])
        self.assertEqual(self.recover(2.0, {5: (3.5, 3.0)}), [])

    def test_rival_memory_resets_when_the_robot_is_seen_again(self):
        self.recover(1.25, {9: (4.0, 3.0)})
        self.recover(1.4, {1: (3.0, 3.0)})
        self.state.observe_robot(1, 1.4, 3.0, 3.0)
        self.recover(1.6, {})
        result = self.recover(2.0, {5: (3.5, 3.0)})
        self.assertEqual([(r.robot_id, r.new_track_id) for r in result], [(1, 5)])

    def test_out_of_field_first_sighting_is_never_a_candidate(self):
        self.recover(1.25, {5: None})
        self.assertEqual(self.recover(2.0, {5: (3.5, 3.0)}), [])

    def test_robot_missing_less_than_the_minimum_is_not_yet_recovered(self):
        self.assertEqual(self.recover(1.25, {5: (3.5, 3.0)}), [])
        result = self.recover(1.5, {5: (3.5, 3.0)})
        self.assertEqual([(r.robot_id, r.old_track_id, r.new_track_id) for r in result], [(1, 1, 5)])
        self.assertAlmostEqual(result[0].distance_ft, 0.5)
        self.assertAlmostEqual(result[0].absence_s, 0.5)
        self.assertAlmostEqual(result[0].gate_ft, 15 * 0.5 + 1.0)

    def test_robot_with_unknown_last_position_is_not_recovered(self):
        state = IdentityRecovery()
        state.note_assignment(1)
        state.observe_tracks(2.0, {5: (3.0, 3.0)})
        self.assertEqual(state.recover(2.0, {5: (3.0, 3.0)}, {1: 1}), [])

    def test_expiry_is_permanent_even_if_the_robot_is_seen_again(self):
        self.assertEqual(self.recover(11.5, {}), [])
        # Its own track comes back briefly, then it is lost once more.
        self.recover(12.0, {1: (3.0, 3.0)})
        self.state.observe_robot(1, 12.0, 3.0, 3.0)
        self.recover(12.2, {})
        self.assertEqual(self.recover(12.6, {5: (3.2, 3.0)}), [])

    def test_previously_assigned_track_is_not_a_candidate(self):
        self.state.note_assignment(5)
        self.assertEqual(self.recover(2.0, {5: (3.5, 3.0)}), [])

    def test_gate_grows_with_elapsed_time(self):
        rules = RecoveryRules()
        self.assertAlmostEqual(rules.gate_ft(0.0), 1.0)
        self.assertAlmostEqual(rules.gate_ft(0.2), 4.0)
        self.assertAlmostEqual(rules.gate_ft(-1.0), 1.0)

    def test_expired_robot_stays_lost_even_if_a_clean_candidate_appears(self):
        self.assertEqual(self.recover(11.5, {}), [])
        self.assertIn(1, self.state.expired)
        self.assertEqual(self.recover(11.6, {5: (3.0, 3.0)}), [])

    def test_each_of_two_lost_robots_with_its_own_clean_candidate_is_recovered(self):
        state = IdentityRecovery()
        for robot_id, track_id, x in ((1, 1, 1.0), (2, 2, 11.0)):
            state.note_assignment(track_id)
            state.observe_robot(robot_id, 1.0, x, 1.0)
        state.observe_tracks(1.05, {5: (1.5, 1.0), 6: (10.5, 1.0)})
        # At 1.5 s each gate (8.5 ft) still excludes the other robot's candidate.
        result = state.recover(1.5, {5: (1.5, 1.0), 6: (10.5, 1.0)}, {1: 1, 2: 2})
        self.assertEqual(sorted((r.robot_id, r.new_track_id) for r in result), [(1, 5), (2, 6)])


if __name__ == "__main__":
    unittest.main()
