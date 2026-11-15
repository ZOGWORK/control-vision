import logging
import math
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.pipeline import manual_tracking
from control_vision.pipeline.manual_tracking import EMIT_THRESHOLD, _offer_photo, _save_best_photos, track_seeded_video

W, H, FPS = 640, 480, 16
CORNERS = [[40, 40], [600, 40], [600, 440], [40, 440]]
COLOURS = {"red": (0, 0, 200), "blue": (200, 70, 0)}


def px_to_ft(x, y):
    return (x - 40) / 560 * 12, (y - 40) / 400 * 12


def lerp(start, end, frames):
    def position(f):
        s = min(1.0, max(0.0, f / max(1, frames - 1)))
        return start[0] + (end[0] - start[0]) * s, start[1] + (end[1] - start[1]) * s
    return position


@dataclass
class Robot:
    alliance: str
    path: Callable[[int], tuple[float, float] | None]
    width: int = 80
    height: int = 70
    flag_on_top: bool = False
    texture_seed: int = 0


def background(seed, tape=False):
    rng = np.random.default_rng(seed)
    tiles = rng.integers(95, 130, (H // 40 + 1, W // 40 + 1))
    gray = np.kron(tiles, np.ones((40, 40)))[:H, :W] + rng.integers(-25, 25, (H, W))
    frame = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    if tape:
        cv2.line(frame, (40, 330), (600, 330), COLOURS["red"], 6)
    return frame


def draw_robot(frame, robot, bottom):
    """Draw a textured robot whose bottom-centre (ground contact) is ``bottom``."""
    bx, by = bottom
    bumper = max(8, robot.height // 4)
    x0, y0 = int(round(bx - robot.width / 2)), int(round(by - robot.height))
    rng = np.random.default_rng(100 + robot.texture_seed)
    body = rng.integers(30, 230, (robot.height - bumper, robot.width)).astype(np.uint8)
    body = cv2.resize(cv2.resize(body, (robot.width // 4, (robot.height - bumper) // 4)),
                      (robot.width, robot.height - bumper), interpolation=cv2.INTER_NEAREST)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + robot.width), min(H, y0 + robot.height - bumper)
    if xs1 > xs0 and ys1 > ys0:
        frame[ys0:ys1, xs0:xs1] = body[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0, None]
    if robot.flag_on_top:
        frame[max(0, y0 + robot.height - bumper):max(0, y0 + robot.height), max(0, x0):max(0, x0 + robot.width)] = 90
        cv2.rectangle(frame, (x0 + robot.width // 3, y0), (x0 + 2 * robot.width // 3, y0 + 12),
                      COLOURS[robot.alliance], -1)
    else:
        cv2.rectangle(frame, (x0, y0 + robot.height - bumper), (x0 + robot.width - 1, y0 + robot.height - 1),
                      COLOURS[robot.alliance], -1)


def write_scenario(path, frame_count, robots, seed=1, tape=False, overlay=None):
    base = background(seed, tape)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    try:
        for f in range(frame_count):
            frame = base.copy()
            for robot in robots:
                bottom = robot.path(f)
                if bottom is not None:
                    draw_robot(frame, robot, bottom)
            if overlay is not None:
                overlay(f, frame)
            writer.write(frame)
    finally:
        writer.release()
    return path


def own_errors(rows, robots_by_id):
    errors = []
    for rid, frame, _, x_ft, y_ft, _ in rows:
        truth = robots_by_id[rid].path(frame)
        if truth is None:
            errors.append(math.inf)
        else:
            tx, ty = px_to_ft(*truth)
            errors.append(math.hypot(x_ft - tx, y_ft - ty))
    return errors


class Capture:
    """In-memory stand-in for cv2.VideoCapture."""

    def __init__(self, frames, fps=FPS, bad=None):
        self.frames = frames
        self.fps = fps
        self.bad = bad or {}
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: len(self.frames)}[prop]

    def grab(self):
        index = self.index
        self.index += 1
        return self.bad.get(index) != "fail"

    def read(self):
        index = self.index
        self.index += 1
        kind = self.bad.get(index)
        if kind == "fail":
            return False, None
        if kind == "gray":
            return True, np.zeros((10, 10), np.uint8)
        return True, self.frames[index]

    def release(self):
        self.released = True


class SeededPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.logger = logging.getLogger("seeded-preview-test")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def run_scenario(self, name, frame_count, robots, ids=None, sample_fps=8, photos=False, seeds=None, **kw):
        path = write_scenario(self.folder / f"{name}.avi", frame_count, robots, **kw)
        ids = ids or list(range(1, len(robots) + 1))
        if seeds is None:
            seeds = [(rid, int(round(robot.path(0)[0])), int(round(robot.path(0)[1]))) for rid, robot in zip(ids, robots)]
        alliances = {rid: robot.alliance for rid, robot in zip(ids, robots)}
        progress = []
        with self.assertLogs(self.logger, level="INFO") as logs:
            rows, photo_rows, events = track_seeded_video(
                path, CORNERS, seeds, alliances, sample_fps, lambda: False,
                lambda *args: progress.append(args), self.logger,
                self.folder / f"{name}-photos" if photos else None)
        self.assertEqual(events, [])
        return rows, photo_rows, logs.output, progress, dict(zip(ids, robots))

    def test_single_robot_projects_clicked_ground_contact(self):
        robot = Robot("red", lerp((150, 300), (450, 380), 40))
        rows, _, _, _, robots = self.run_scenario("single", 40, [robot])
        self.assertGreaterEqual(len(rows), 16)
        self.assertLess(max(own_errors(rows, robots)), 0.25)

    def test_colour_on_top_does_not_move_projected_point(self):
        robot = Robot("red", lerp((150, 300), (450, 380), 40), flag_on_top=True)
        rows, _, _, _, robots = self.run_scenario("flag-top", 40, [robot])
        self.assertGreaterEqual(len(rows), 16)
        self.assertLess(max(own_errors(rows, robots)), 0.25)

    def test_same_alliance_crossing_never_swaps_identity(self):
        first = Robot("red", lerp((150, 300), (500, 300), 48), texture_seed=1)
        second = Robot("red", lerp((500, 300), (150, 300), 48), texture_seed=2)
        rows, _, logs, _, robots = self.run_scenario("cross-rr", 48, [first, second])
        errors = own_errors(rows, robots)
        self.assertLess(max(errors), 0.6, "a row was assigned to the wrong robot")
        self.assertTrue(any("suspended" in line and "same-alliance" in line for line in logs))
        for rid in (1, 2):
            frames = [row[1] for row in rows if row[0] == rid]
            self.assertTrue(any(f < 12 for f in frames) and any(f > 36 for f in frames),
                            f"robot {rid} should be tracked before and after the crossing")

    def test_unseeded_same_alliance_robot_does_not_take_over_track(self):
        tracked = Robot("red", lerp((150, 300), (500, 300), 48), texture_seed=1)
        passer = Robot("red", lerp((500, 300), (150, 300), 48), texture_seed=2)
        path = write_scenario(self.folder / "unseeded.avi", 48, [tracked, passer])
        with self.assertLogs(self.logger, level="INFO") as logs:
            rows, _, _ = track_seeded_video(path, CORNERS, [(1, 150, 300)], {1: "red"}, 8, lambda: False,
                                            lambda *_: None, self.logger)
        self.assertTrue(rows)
        self.assertLess(max(own_errors(rows, {1: tracked})), 0.6, "the track followed the other robot")
        self.assertTrue(any("robot-sized blob" in line for line in logs.output))

    def test_static_alliance_tape_does_not_capture_track(self):
        robot = Robot("red", lambda f: (300, 360 - 9 * f) if 360 - 9 * f > -80 else None)
        rows, _, _, _, robots = self.run_scenario("tape", 48, [robot], tape=True)
        self.assertGreater(len(rows), 4)
        # Every row is on the robot: none stays on the tape after the robot
        # drives off it, and none once the robot has left the frame.
        self.assertLess(max(own_errors(rows, robots)), 0.5)
        tape_y_ft = px_to_ft(0, 330)[1]
        self.assertFalse([row for row in rows if abs(row[4] - tape_y_ft) < 0.5 and robot.path(row[1])[1] < 300])

    def test_occluded_robot_is_reacquired_with_reduced_confidence(self):
        robot = Robot("red", lerp((120, 300), (520, 300), 48))

        def occluder(f, frame):
            if 14 <= f < 30:
                cv2.rectangle(frame, (250, 180), (400, 320), (60, 60, 60), -1)

        rows, _, logs, _, robots = self.run_scenario("occlude", 48, [robot], overlay=occluder)
        self.assertLess(max(own_errors(rows, robots)), 0.5)
        self.assertTrue(any("reacquired" in line for line in logs))
        after = [row for row in rows if row[1] >= 30]
        self.assertGreaterEqual(len(after), 4, "track should resume after the robot reappears")
        self.assertLess(after[0][5], 0.6, "first point after a long gap must not be fully confident")
        self.assertGreater(max(row[5] for row in after), 0.6)

    def test_lost_robot_is_not_emitted_at_last_position(self):
        robot = Robot("red", lambda f: (300, 300) if f < 20 else None)
        rows, _, logs, _, _ = self.run_scenario("lost", 40, [robot])
        self.assertTrue(rows)
        self.assertFalse([row for row in rows if row[1] >= 20])
        self.assertTrue(any("suspended" in line for line in logs))

    def test_hidden_alliance_colour_keeps_points_below_threshold(self):
        robot = Robot("red", lerp((150, 300), (450, 300), 40))

        def strip(f, frame):
            if 14 <= f < 26:
                cv2.rectangle(frame, (0, 280), (W - 1, 302), (60, 60, 60), -1)

        rows, _, _, _, _ = self.run_scenario("bumper-hidden", 40, [robot], overlay=strip)
        self.assertTrue([row for row in rows if row[1] < 14])
        self.assertFalse([row for row in rows if 14 <= row[1] < 26])
        self.assertTrue(all(row[5] >= EMIT_THRESHOLD for row in rows))

    def test_hidden_flag_keeps_points_below_threshold_even_when_robot_matches(self):
        # The flag is a small part of the robot, so the template still matches
        # while it is hidden; missing alliance colour alone must cap confidence.
        robot = Robot("red", lerp((150, 300), (450, 300), 40), flag_on_top=True)

        def cover(f, frame):
            if 14 <= f < 26:
                cv2.rectangle(frame, (0, 228), (W - 1, 244), (60, 60, 60), -1)

        rows, _, _, _, _ = self.run_scenario("flag-hidden", 40, [robot], overlay=cover)
        self.assertTrue([row for row in rows if row[1] < 14])
        self.assertFalse([row for row in rows if 14 <= row[1] < 26])

    def test_points_outside_the_calibrated_field_are_dropped(self):
        robot = Robot("blue", lerp((400, 300), (640, 300), 40))
        rows, _, _, _, _ = self.run_scenario("leave-field", 40, [robot])
        self.assertTrue(rows)
        self.assertTrue(all(0 <= row[3] <= 12 and 0 <= row[4] <= 12 for row in rows))

    def test_sample_rate_and_progress(self):
        robot = Robot("red", lerp((150, 300), (300, 300), 41))
        for sample_fps, stride in ((8, 2), (5, 3)):
            rows, _, _, progress, _ = self.run_scenario(f"rate-{sample_fps}", 41, [robot], sample_fps=sample_fps)
            self.assertEqual(len(progress), math.ceil(41 / stride))
            percents = [item[1] for item in progress]
            self.assertEqual(percents, sorted(set(percents)))
            self.assertAlmostEqual(percents[-1], 100.0)
            self.assertTrue(rows)
            self.assertTrue(all(row[1] % stride == 0 for row in rows))

    def test_photo_crop_scales_with_robot_and_small_robot_is_skipped(self):
        sizes = {}
        for width in (200, 260):
            robot = Robot("red", lerp((320, 300), (330, 300), 16), width=width, height=int(width * 0.75))
            _, photos, _, _, _ = self.run_scenario(f"photo-{width}", 16, [robot], photos=True)
            self.assertEqual(len(photos), 1)
            image = cv2.imread(str(photos[0][1]))
            sizes[width] = image.shape[1]
            self.assertGreaterEqual(min(image.shape[:2]), 200)
        self.assertAlmostEqual(sizes[260] / sizes[200], 1.3, delta=0.12)
        self.assertAlmostEqual(sizes[200] / 200, 1.4, delta=0.15)
        robot = Robot("red", lerp((320, 300), (330, 300), 16))
        _, photos, logs, _, _ = self.run_scenario("photo-small", 16, [robot], photos=True)
        self.assertEqual(photos, [])
        self.assertTrue(any("photo skipped" in line and "below 200x200" in line for line in logs))

    def test_best_photo_is_largest_confident_box(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        best = {}
        _offer_photo(best, 1, frame, (100, 100, 300, 300), 0.9, 0.0)   # seed frame, score 36000
        _offer_photo(best, 1, frame, (100, 100, 360, 340), 0.8, 1.0)   # larger box, score 49920
        _offer_photo(best, 1, frame, (100, 100, 420, 400), 0.2, 2.0)   # largest but weak, 19200
        _offer_photo(best, 1, frame, (100, 100, 200, 200), 0.9, 3.0)   # small, 9000
        self.assertEqual(best[1][2], 1.0)
        self.assertEqual(best[1][3].shape[:2], (336, 364))

    def test_cancel_raises_and_releases_capture(self):
        path = write_scenario(self.folder / "cancel.avi", 30, [Robot("red", lerp((150, 300), (300, 300), 30))])
        captures = []
        real = cv2.VideoCapture

        def spy(*args):
            capture = real(*args)
            captures.append(capture)
            return capture

        calls = []
        with patch.object(manual_tracking.cv2, "VideoCapture", side_effect=spy):
            with self.assertRaises(InterruptedError):
                track_seeded_video(path, CORNERS, [(1, 150, 300)], {1: "red"}, 8,
                                   lambda: calls.append(1) or len(calls) > 3, lambda *_: None, self.logger)
        self.assertEqual(len(calls), 4)
        self.assertFalse(captures[0].isOpened())

    def test_unreadable_file_names_the_file(self):
        junk = self.folder / "broken-match.mp4"
        junk.write_bytes(b"not a video" * 500)
        for path in (junk, self.folder / "missing-match.mp4"):
            with self.assertRaisesRegex(RuntimeError, f"Could not open video file {path.name}"):
                track_seeded_video(path, CORNERS, [(1, 150, 300)], {1: "red"}, 8,
                                   lambda: False, lambda *_: None, self.logger)

    def frames_for(self, count=24):
        robot = Robot("red", lerp((150, 300), (330, 300), count))
        base = background(3)
        frames = []
        for f in range(count):
            frame = base.copy()
            draw_robot(frame, robot, robot.path(f))
            frames.append(frame)
        return frames

    def run_fake(self, capture, seeds=((1, 150, 300),), alliances=None, sample_fps=8):
        with patch.object(manual_tracking.cv2, "VideoCapture", return_value=capture):
            return track_seeded_video(Path("match.mp4"), CORNERS, list(seeds), alliances or {1: "red"}, sample_fps,
                                      lambda: False, lambda *_: None, self.logger)

    def test_failed_reads_are_skipped_then_stop_after_repeated_failures(self):
        capture = Capture(self.frames_for(), bad={8: "fail", 9: "fail"})
        with self.assertLogs(self.logger, level="WARNING") as logs:
            rows, _, _ = self.run_fake(capture)
        self.assertTrue(capture.released)
        self.assertTrue(any(row[1] > 10 for row in rows), "tracking should continue after one bad frame")
        self.assertTrue(any("Could not read frame 8" in line for line in logs.output))
        self.assertTrue(any("Could not grab frame 9" in line for line in logs.output))
        capture = Capture(self.frames_for(), bad={f: "fail" for f in range(6, 24)})
        with self.assertLogs(self.logger, level="WARNING") as logs:
            rows, _, _ = self.run_fake(capture)
        self.assertTrue(rows)
        self.assertTrue(all(row[1] < 6 for row in rows))
        self.assertTrue(any("consecutive unreadable frames" in line for line in logs.output))
        self.assertLess(capture.index, 24)

    def test_malformed_frame_is_skipped(self):
        capture = Capture(self.frames_for(), bad={10: "gray"})
        with self.assertLogs(self.logger, level="WARNING") as logs:
            rows, _, _ = self.run_fake(capture)
        self.assertTrue(any("Skipped frame 10" in line for line in logs.output))
        self.assertTrue(any(row[1] > 10 for row in rows))
        self.assertTrue(capture.released)

    def test_input_validation_raises_plain_errors_and_releases_video(self):
        frames = self.frames_for(4)
        cases = [
            ({"alliances": {1: "green"}}, "not red or blue"),
            ({"alliances": {2: "red"}}, "has no alliance"),
            ({"seeds": ((1, 700, 300),)}, "outside the 640x480 video frame"),
            ({"sample_fps": 4}, "between 5 and 10"),
            ({"sample_fps": 11}, "between 5 and 10"),
            ({"sample_fps": float("nan")}, "between 5 and 10"),
        ]
        for kwargs, message in cases:
            capture = Capture(frames)
            with self.subTest(message=message, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_fake(capture, **kwargs)
                self.assertTrue(capture.released)
        rows, _, _ = self.run_fake(Capture(frames), alliances={1: " Red "})
        self.assertTrue(rows)

    def test_invalid_calibration_releases_video(self):
        capture = Capture(self.frames_for(4))
        with patch.object(manual_tracking.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "four field corners"):
                track_seeded_video(Path("match.mp4"), [], [(1, 50, 50)], {1: "red"}, 8,
                                   lambda: False, lambda *_: None, self.logger)
        self.assertTrue(capture.released)

    def test_only_full_size_robot_photos_are_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = _save_best_photos({
                1: (1, 1.0, 0.0, np.zeros((199, 250, 3), dtype=np.uint8)),
                2: (1, 1.0, 0.0, np.zeros((200, 200, 3), dtype=np.uint8)),
            }, Path(folder), self.logger)
            self.assertEqual([row[0] for row in rows], [2])
            self.assertFalse((Path(folder) / "robot-1.png").exists())
            self.assertTrue((Path(folder) / "robot-2.png").is_file())


if __name__ == "__main__":
    unittest.main()
