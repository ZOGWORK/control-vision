import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.pipeline.manual_tracking import _photo_crop_bounds, _save_best_photos, track_seeded_video


class SeededPreviewTests(unittest.TestCase):
    def test_seeded_robot_produces_field_samples_on_moving_frames(self):
        frames = []
        for offset in range(6):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            left = 140 + 3 * offset
            cv2.rectangle(frame, (left, 200), (left + 80, 300), (0, 0, 200), -1)
            for x in range(left + 8, left + 75, 16):
                for y in range(210, 290, 16):
                    cv2.rectangle(frame, (x, y), (x + 4, y + 4), (255, 255, 255), -1)
            frames.append(frame)

        class Capture:
            index = 0
            released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 5, cv2.CAP_PROP_FRAME_COUNT: len(frames)}[prop]
            def read(self):
                frame = frames[self.index]
                self.index += 1
                return True, frame
            def release(self): self.released = True

        capture = Capture()
        with patch("control_vision.pipeline.manual_tracking.cv2.VideoCapture", return_value=capture):
            rows, photos, events = track_seeded_video(
                Path("match.mp4"), [[0, 0], [639, 0], [639, 479], [0, 479]],
                [(1, 180, 300)], {1: "red"}, 5, lambda: False,
                lambda *_: None, logging.getLogger("preview-motion-test"),
            )
        self.assertTrue(capture.released)
        self.assertGreaterEqual(len(rows), 4)
        self.assertTrue(all(row[0] == 1 and 0 <= row[3] <= 12 and 0 <= row[4] <= 12 for row in rows))
        self.assertEqual(photos, [])
        self.assertEqual(events, [])

    def test_invalid_calibration_releases_video(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 100}[prop]
            def release(self): self.released = True

        capture = Capture()
        with patch("control_vision.pipeline.manual_tracking.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "four field corners"):
                track_seeded_video(Path("match.mp4"), [], [(1, 50, 50)], {1: "red"}, 8,
                                   lambda: False, lambda *_: None, logging.getLogger("preview-test"))
        self.assertTrue(capture.released)

    def test_zero_sample_rate_releases_video(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 100}[prop]
            def release(self): self.released = True

        capture = Capture()
        with patch("control_vision.pipeline.manual_tracking.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "Sample frame rate"):
                track_seeded_video(Path("match.mp4"), [], [(1, 50, 50)], {1: "red"}, 0,
                                   lambda: False, lambda *_: None, logging.getLogger("preview-test"))
        self.assertTrue(capture.released)

    def test_only_full_size_robot_photos_are_exported(self):
        x0, y0, x1, y1 = _photo_crop_bounds((720, 1280, 3), 600, 510, 112, 88)
        self.assertGreaterEqual(x1 - x0, 200)
        self.assertGreaterEqual(y1 - y0, 200)
        with tempfile.TemporaryDirectory() as folder:
            rows = _save_best_photos({
                1: (1, 1.0, 0.0, np.zeros((199, 250, 3), dtype=np.uint8)),
                2: (1, 1.0, 0.0, np.zeros((200, 200, 3), dtype=np.uint8)),
            }, Path(folder), logging.getLogger("preview-photo-test"))
            self.assertEqual([row[0] for row in rows], [2])
            self.assertFalse((Path(folder) / "robot-1.png").exists())
            self.assertTrue((Path(folder) / "robot-2.png").is_file())


if __name__ == "__main__":
    unittest.main()
