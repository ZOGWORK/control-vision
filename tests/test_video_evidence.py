import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.pipeline.homography import matrix_from_corners, project_point
from control_vision.pipeline.ingest import VideoError, probe_video
from control_vision.pipeline.scoreboard_ocr import _consensus, _read_digits, read_scoreboard
from control_vision.pipeline.clips import create_highlight


class VideoEvidenceTests(unittest.TestCase):
    def test_stream_duration_falls_back_to_readable_frame_count(self):
        class Capture:
            released = False
            def get(self, prop): return {cv2.CAP_PROP_FRAME_COUNT: 300}[prop]
            def release(self): self.released = True

        capture = Capture()
        result = SimpleNamespace(returncode=0, stdout=(
            '{"streams":[{"avg_frame_rate":"30/1","duration":"N/A",'
            '"width":1280,"height":720}]}'
        ))
        with patch("control_vision.pipeline.ingest.shutil.which", return_value="ffprobe"), \
             patch("control_vision.pipeline.ingest.subprocess.run", return_value=result), \
             patch("control_vision.pipeline.ingest.cv2.VideoCapture", return_value=capture):
            metadata = probe_video(Path("match.mp4"))
        self.assertEqual(metadata["duration_s"], 10)
        self.assertTrue(capture.released)

    def test_invalid_ffprobe_rate_does_not_import_video(self):
        result = SimpleNamespace(returncode=0, stdout=(
            '{"streams":[{"avg_frame_rate":"0/1","duration":"10",'
            '"width":1280,"height":720}]}'
        ))
        with patch("control_vision.pipeline.ingest.shutil.which", return_value="ffprobe"), \
             patch("control_vision.pipeline.ingest.subprocess.run", return_value=result):
            with self.assertRaisesRegex(VideoError, "match.mp4"):
                probe_video(Path("match.mp4"))

    def test_fallback_rejects_video_with_zero_frames(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 0,
                        cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720}[prop]
            def release(self): self.released = True

        capture = Capture()
        with patch("control_vision.pipeline.ingest.shutil.which", return_value=None), \
             patch("control_vision.pipeline.ingest.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(VideoError, "match.mp4"):
                probe_video(Path("match.mp4"))
        self.assertTrue(capture.released)

    def test_homography_maps_field_corners_and_rejects_degenerate_quad(self):
        matrix = matrix_from_corners([[10, 10], [130, 10], [130, 130], [10, 130]])
        self.assertEqual(tuple(round(value, 4) for value in project_point(matrix, 70, 70)), (6.0, 6.0))
        with self.assertRaises(ValueError):
            matrix_from_corners([[10, 10], [10, 10], [10, 10], [10, 10]])

    def test_ocr_consensus_retains_earlier_high_score_and_masks_polygon(self):
        result = _consensus([{"t_s": 1, "value": 100, "confidence": .9},
                             {"t_s": 2, "value": 90, "confidence": .9},
                             {"t_s": 3, "value": 90, "confidence": .9}])
        self.assertEqual(result["value"], 100)
        self.assertTrue(result["later_lower_reading"])

        class Engine:
            crop = None
            def __call__(self, image):
                self.crop = image.copy()
                return []

        engine = Engine()
        _read_digits(engine, np.full((20, 20, 3), 255, dtype=np.uint8), [[0, 0], [19, 0], [0, 19]])
        self.assertIsNotNone(engine.crop)
        self.assertEqual(int(engine.crop[-1, -1].max()), 0)

    def test_ocr_consensus_uses_later_stable_high_score(self):
        rows = ([{"t_s": float(i), "value": 90, "confidence": .99} for i in range(8)] +
                [{"t_s": float(i), "value": 100, "confidence": .99} for i in range(8, 12)])
        self.assertEqual(_consensus(rows)["value"], 100)

    def test_ocr_engine_failure_releases_video(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 300}[prop]
            def release(self): self.released = True

        capture = Capture()
        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=capture), \
             patch("rapidocr_onnxruntime.RapidOCR", side_effect=RuntimeError("model load failed")):
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                read_scoreboard(Path("match.mp4"), [], [])
        self.assertTrue(capture.released)

    def test_one_bad_ocr_frame_does_not_discard_later_evidence(self):
        class Capture:
            released = False
            reads = 0
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 1, cv2.CAP_PROP_FRAME_COUNT: 3}[prop]
            def set(self, *_): pass
            def read(self):
                self.reads += 1
                return True, np.zeros((20, 20, 3), dtype=np.uint8)
            def release(self): self.released = True

        capture = Capture()
        ocr_values = [RuntimeError("one frame failed"), (100, .9, "100"), (90, .9, "90"),
                      (100, .9, "100"), (90, .9, "90")]
        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=capture), \
             patch("rapidocr_onnxruntime.RapidOCR"), \
             patch("control_vision.pipeline.scoreboard_ocr._read_digits", side_effect=ocr_values), \
             patch("control_vision.pipeline.scoreboard_ocr.logging.getLogger"):
            result = read_scoreboard(Path("match.mp4"), [], [], sample_fps=1, tail_seconds=3)
        self.assertEqual(result["frames_sampled"], 2)
        self.assertEqual(result["red"]["value"], 100)
        self.assertTrue(capture.released)

    def test_failed_clip_copy_reencodes_without_leaving_partial_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "clip.mp4"
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    output.write_bytes(b"partial")
                    return SimpleNamespace(returncode=1, stderr="copy failed")
                self.assertFalse(output.exists())
                output.write_bytes(b"encoded")
                return SimpleNamespace(returncode=0, stderr="")

            with patch("control_vision.pipeline.clips.subprocess.run", side_effect=run):
                self.assertEqual(create_highlight(Path("match.mp4"), output, 1.0, "ffmpeg"), (0, 4.0))
            self.assertEqual(output.read_bytes(), b"encoded")
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
