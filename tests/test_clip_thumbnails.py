"""Clip thumbnails are decoded once and cached beside the clip."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.pipeline.clips import thumbnail, thumbnail_cache_path


def _write_clip(path: Path, colour: int, frames: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    for _ in range(frames):
        writer.write(np.full((360, 640, 3), colour, dtype=np.uint8))
    writer.release()


class ClipThumbnailTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.clip = Path(folder.name) / "event-1.mp4"
        _write_clip(self.clip, 200)

    def test_first_call_decodes_and_caches_a_small_thumbnail(self):
        frame = thumbnail(self.clip)
        self.assertEqual(frame.shape[:2], (135, 240))
        self.assertTrue(thumbnail_cache_path(self.clip).is_file())
        self.assertAlmostEqual(float(frame.mean()), 200, delta=6)

    def test_later_calls_use_the_cache_without_opening_the_clip(self):
        thumbnail(self.clip)
        with patch("control_vision.pipeline.clips.cv2.VideoCapture", side_effect=AssertionError("decoded")):
            frame = thumbnail(self.clip)
        self.assertEqual(frame.shape[:2], (135, 240))

    def test_a_regenerated_clip_gets_a_new_thumbnail(self):
        thumbnail(self.clip)
        cache = thumbnail_cache_path(self.clip)
        old = time.time() - 60
        os.utime(cache, (old, old))
        _write_clip(self.clip, 30)
        frame = thumbnail(self.clip)
        self.assertAlmostEqual(float(frame.mean()), 30, delta=6)
        self.assertGreaterEqual(cache.stat().st_mtime_ns, self.clip.stat().st_mtime_ns)

    def test_an_unreadable_clip_has_no_thumbnail_and_no_cache(self):
        broken = self.clip.with_name("broken.mp4")
        broken.write_bytes(b"not a video")
        self.assertIsNone(thumbnail(broken))
        self.assertFalse(thumbnail_cache_path(broken).exists())

    def test_a_corrupt_cache_is_replaced(self):
        cache = thumbnail_cache_path(self.clip)
        cache.write_bytes(b"not a png")
        frame = thumbnail(self.clip)
        self.assertEqual(frame.shape[:2], (135, 240))
        self.assertIsNotNone(cv2.imread(str(cache)))


if __name__ == "__main__":
    unittest.main()
