"""Image files under non-ASCII Windows paths (OpenCV's own imread/imwrite fail there)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.image_files import read_image, write_image

UNICODE_FOLDER = "Équipe ñandú 机器人"
ROOT = Path(__file__).resolve().parents[1]


def _run_fresh(script: str, *args: str) -> subprocess.CompletedProcess:
    """Run in a new interpreter: importing Ultralytics anywhere in a shared test
    process replaces cv2.imwrite/imread with Unicode-safe versions and would
    hide the bug these tests check."""
    return subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True,
                          encoding="utf-8", cwd=ROOT, timeout=180,
                          env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"})


PHOTO_SCRIPT = """
import logging, sys
from pathlib import Path
import numpy as np
from control_vision.pipeline.manual_tracking import _save_best_photos
crop = np.full((240, 260, 3), 90, dtype=np.uint8)
rows = _save_best_photos({1: (1.0, 0.9, 2.0, crop)}, Path(sys.argv[1]), logging.getLogger("t"))
assert "ultralytics" not in sys.modules
print(len(rows), rows[0][1].is_file() if rows else False)
"""

EXTRACT_SCRIPT = """
import sys
from pathlib import Path
import cv2
import numpy as np
from control_vision.training.extract_frames import extract
video = Path(sys.argv[1])
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
for value in range(30):
    writer.write(np.full((48, 64, 3), value * 8, dtype=np.uint8))
writer.release()
written = extract(video, Path(sys.argv[2]), every_seconds=1.0)
assert "ultralytics" not in sys.modules
print(written, len(list(Path(sys.argv[2]).rglob("*.jpg"))))
"""


class ImageFileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / UNICODE_FOLDER
        self.folder.mkdir()
        self.image = np.zeros((40, 60, 3), dtype=np.uint8)
        self.image[:, :30] = (10, 200, 30)

    def test_png_round_trip_in_a_non_ascii_folder(self):
        path = self.folder / "robot-1.png"
        self.assertTrue(write_image(path, self.image))
        self.assertTrue(np.array_equal(read_image(path), self.image))

    def test_jpeg_with_quality_in_a_non_ascii_folder(self):
        path = self.folder / "frame.jpg"
        self.assertTrue(write_image(path, self.image, [cv2.IMWRITE_JPEG_QUALITY, 95]))
        self.assertLess(float(np.abs(read_image(path).astype(int) - self.image.astype(int)).mean()), 3)

    def test_failures_return_false_or_none_and_leave_no_file(self):
        missing_dir = self.folder / "missing" / "robot.png"
        self.assertFalse(write_image(missing_dir, self.image))
        self.assertFalse(missing_dir.exists())
        unknown = self.folder / "robot.notanimage"
        self.assertFalse(write_image(unknown, self.image))
        self.assertFalse(unknown.exists())
        with patch("pathlib.Path.write_bytes", side_effect=PermissionError("read-only")):
            self.assertFalse(write_image(self.folder / "locked.png", self.image))
        with patch("pathlib.Path.write_bytes", side_effect=PermissionError("disk full")), \
                patch("pathlib.Path.unlink", side_effect=PermissionError("locked")):
            self.assertFalse(write_image(self.folder / "stuck.png", self.image))
        self.assertFalse((self.folder / "locked.png").exists())
        self.assertIsNone(read_image(self.folder / "absent.png"))
        garbage = self.folder / "garbage.png"
        garbage.write_bytes(b"not an image")
        self.assertIsNone(read_image(garbage))
        empty = self.folder / "empty.png"
        empty.write_bytes(b"")
        self.assertIsNone(read_image(empty))

    def test_seeded_preview_saves_robot_photos_in_a_non_ascii_project(self):
        result = _run_fresh(PHOTO_SCRIPT, str(self.folder / "photos"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["1", "True"])

    def test_frame_extraction_into_a_non_ascii_folder(self):
        result = _run_fresh(EXTRACT_SCRIPT, str(Path(tempfile.mkdtemp()) / "match.mp4"),
                            str(self.folder / "frames"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["3", "3"])


if __name__ == "__main__":
    unittest.main()
