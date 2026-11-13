import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.training.extract_frames import extract


class FrameExtractionTests(unittest.TestCase):
    def test_zero_limit_is_rejected_before_video_open(self):
        with patch("control_vision.training.extract_frames.cv2.VideoCapture") as video:
            with self.assertRaisesRegex(ValueError, "--max-frames"):
                extract(Path("match.mp4"), Path("unused"), max_frames=0)
            video.assert_not_called()

    def test_early_decode_failure_names_video_and_releases_capture(self):
        class Capture:
            def __init__(self, valid_frames):
                self.valid_frames = valid_frames
                self.reads = 0
                self.released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 10}[prop]
            def read(self):
                self.reads += 1
                return self.reads <= self.valid_frames, np.zeros((8, 8, 3), dtype=np.uint8)
            def release(self): self.released = True

        for valid_frames in (2, 9):
            with self.subTest(valid_frames=valid_frames), tempfile.TemporaryDirectory() as folder:
                video = Capture(valid_frames)
                with patch("control_vision.training.extract_frames.cv2.VideoCapture", return_value=video), \
                     patch("control_vision.training.extract_frames.cv2.imwrite", return_value=True):
                    with self.assertRaisesRegex(RuntimeError, "match.mp4"):
                        extract(Path("match.mp4"), Path(folder))
                self.assertTrue(video.released)

    def test_complete_video_and_explicit_frame_limit(self):
        class Capture:
            def __init__(self):
                self.reads = 0
                self.released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 2, cv2.CAP_PROP_FRAME_COUNT: 4}[prop]
            def read(self):
                self.reads += 1
                return self.reads <= 4, np.zeros((8, 8, 3), dtype=np.uint8)
            def release(self): self.released = True

        with tempfile.TemporaryDirectory() as folder:
            for limit, expected in ((None, 4), (2, 2)):
                with self.subTest(limit=limit):
                    video = Capture()
                    with patch("control_vision.training.extract_frames.cv2.VideoCapture", return_value=video), \
                         patch("control_vision.training.extract_frames.cv2.imwrite", return_value=True):
                        self.assertEqual(extract(Path("match.mp4"), Path(folder), every_seconds=.5,
                                                 max_frames=limit), expected)
                    self.assertTrue(video.released)


if __name__ == "__main__":
    unittest.main()
