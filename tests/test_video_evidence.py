import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from control_vision.pipeline.homography import matrix_from_corners, project_point, reprojection_error_px
from control_vision.pipeline.ingest import VideoError, probe_video
from control_vision.pipeline.scoreboard_ocr import _consensus, _read_digits, read_scoreboard
from control_vision.pipeline.clips import create_highlight


ROI = [[1, 1], [18, 1], [18, 18], [1, 18]]


def _write_video(path: Path, frames: int = 90, fps: float = 30.0, size=(160, 120)) -> Path:
    """Write a small synthetic MJPG AVI; each frame carries its index as text."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV cannot write MJPG AVI files on this machine.")
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), index % 250, dtype=np.uint8)
        cv2.putText(frame, str(index), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


def _fake_rapidocr(engine: MagicMock) -> dict:
    """Stand in for the optional RapidOCR package so tests do not need its models."""
    module = ModuleType("rapidocr_onnxruntime")
    module.RapidOCR = engine  # type: ignore[attr-defined]
    return {"rapidocr_onnxruntime": module}


def _ok_run(*_args, **_kwargs):
    return SimpleNamespace(returncode=0, stderr="")


class VideoEvidenceTests(unittest.TestCase):
    def test_stream_duration_falls_back_to_readable_frame_count(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop): return {cv2.CAP_PROP_FRAME_COUNT: 300}[prop]
            def set(self, *_): return True
            def read(self): return True, np.zeros((720, 1280, 3), dtype=np.uint8)
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
        self.assertEqual(metadata["warnings"], [])
        self.assertTrue(capture.released)

    def test_ffprobe_zero_avg_frame_rate_falls_back_to_r_frame_rate(self):
        result = SimpleNamespace(returncode=0, stdout=(
            '{"streams":[{"avg_frame_rate":"0/0","r_frame_rate":"30/1","duration":"10",'
            '"width":1280,"height":720}]}'
        ))
        with patch("control_vision.pipeline.ingest.shutil.which", return_value="ffprobe"),              patch("control_vision.pipeline.ingest.subprocess.run", return_value=result),              patch("control_vision.pipeline.ingest._check_decodable", side_effect=lambda _path, meta: meta):
            metadata = probe_video(Path("match.mkv"))
        self.assertEqual(metadata["fps"], 30.0)

    def test_ffprobe_without_usable_rates_falls_back_to_opencv_fps(self):
        class Capture:
            def isOpened(self): return True
            def get(self, prop): return {cv2.CAP_PROP_FPS: 25.0}[prop]
            def release(self): pass

        result = SimpleNamespace(returncode=0, stdout=(
            '{"streams":[{"avg_frame_rate":"0/0","r_frame_rate":"0/0","duration":"10",'
            '"width":1280,"height":720}]}'
        ))
        with patch("control_vision.pipeline.ingest.shutil.which", return_value="ffprobe"),              patch("control_vision.pipeline.ingest.subprocess.run", return_value=result),              patch("control_vision.pipeline.ingest.cv2.VideoCapture", return_value=Capture()),              patch("control_vision.pipeline.ingest._check_decodable", side_effect=lambda _path, meta: meta):
            metadata = probe_video(Path("match.webm"))
        self.assertEqual(metadata["fps"], 25.0)

    def test_hung_or_unrunnable_ffprobe_falls_back_to_opencv(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            video = _write_video(Path(folder) / "match.avi", frames=60, fps=30.0, size=(1280, 720))
            for failure in (subprocess.TimeoutExpired(["ffprobe"], 30), PermissionError(13, "denied")):
                def run(command, **kwargs):
                    self.assertIn("timeout", kwargs)
                    raise failure
                with patch("control_vision.pipeline.ingest.shutil.which", return_value="ffprobe"),                      patch("control_vision.pipeline.ingest.subprocess.run", side_effect=run),                      self.assertLogs("control_vision.pipeline.ingest", "WARNING"):
                    metadata = probe_video(video)
                self.assertAlmostEqual(metadata["duration_s"], 2.0, places=3)
                self.assertEqual(metadata["fps"], 30.0)

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

    def test_probe_warns_but_does_not_block_low_resolution_or_frame_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _write_video(Path(folder) / "small.avi", frames=40, fps=10.0, size=(160, 120))
            with patch("control_vision.pipeline.ingest.shutil.which", return_value=None):
                metadata = probe_video(video)
        self.assertEqual((metadata["width"], metadata["height"]), (160, 120))
        self.assertEqual(len(metadata["warnings"]), 2)
        self.assertIn("1280×720 or higher is recommended", metadata["warnings"][0])
        self.assertIn("15 fps or higher is recommended", metadata["warnings"][1])

    def test_probe_of_good_video_has_no_warnings(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _write_video(Path(folder) / "good.avi", frames=60, fps=30.0, size=(1280, 720))
            with patch("control_vision.pipeline.ingest.shutil.which", return_value=None):
                metadata = probe_video(video)
        self.assertEqual(metadata["warnings"], [])
        self.assertAlmostEqual(metadata["duration_s"], 2.0, places=3)
        self.assertNotIn("truncated", metadata)

    def test_probe_detects_truncated_video_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _write_video(Path(folder) / "full.avi", frames=300, fps=30.0)
            truncated = Path(folder) / "cut.avi"
            data = video.read_bytes()
            truncated.write_bytes(data[: len(data) // 2])
            with patch("control_vision.pipeline.ingest.shutil.which", return_value=None):
                metadata = probe_video(truncated)
        self.assertTrue(metadata["truncated"])
        self.assertAlmostEqual(metadata["reported_duration_s"], 10.0, places=3)
        self.assertLess(metadata["duration_s"], 7.0)
        self.assertGreater(metadata["duration_s"], 2.0)
        self.assertTrue(any("may be truncated" in warning and "cut.avi" in warning
                            for warning in metadata["warnings"]))

    def test_probe_warns_when_truncated_tail_end_cannot_be_located(self):
        class Capture:
            position = 0
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 300, cv2.CAP_PROP_POS_FRAMES: 0,
                        cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720}[prop]
            def set(self, _prop, value): self.position = int(value)
            def read(self): return (self.position == 0), None
            def release(self): pass

        with patch("control_vision.pipeline.ingest.shutil.which", return_value=None), \
             patch("control_vision.pipeline.ingest.cv2.VideoCapture", return_value=Capture()):
            metadata = probe_video(Path("tail.mp4"))
        self.assertTrue(metadata["truncated"])
        self.assertEqual(metadata["duration_s"], 10.0)
        self.assertEqual(len(metadata["warnings"]), 1)
        self.assertIn("tail.mp4 may be truncated", metadata["warnings"][0])

    def test_probe_rejects_video_whose_first_frame_cannot_be_decoded(self):
        class Capture:
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 300,
                        cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720}[prop]
            def set(self, *_): return True
            def read(self): return False, None
            def release(self): pass

        with patch("control_vision.pipeline.ingest.shutil.which", return_value=None), \
             patch("control_vision.pipeline.ingest.cv2.VideoCapture", return_value=Capture()):
            with self.assertRaisesRegex(VideoError, "first frame of broken.mp4"):
                probe_video(Path("broken.mp4"))

    def test_probe_rejects_unreadable_file_by_name(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _write_video(Path(folder) / "full.avi", frames=30, fps=30.0)
            broken = Path(folder) / "stub.avi"
            broken.write_bytes(video.read_bytes()[:200])
            with patch("control_vision.pipeline.ingest.shutil.which", return_value=None):
                with self.assertRaisesRegex(VideoError, "stub.avi"):
                    probe_video(broken)

    def test_homography_maps_field_corners_and_rejects_degenerate_quad(self):
        matrix = matrix_from_corners([[10, 10], [130, 10], [130, 130], [10, 130]])
        self.assertEqual(tuple(round(value, 4) for value in project_point(matrix, 70, 70)), (6.0, 6.0))
        # A non-centre point distinguishes x from y, so a transposed target order fails.
        self.assertEqual(tuple(round(value, 4) for value in project_point(matrix, 40, 100)), (3.0, 9.0))
        with self.assertRaises(ValueError):
            matrix_from_corners([[10, 10], [10, 10], [10, 10], [10, 10]])

    def test_homography_maps_each_clicked_corner_to_its_field_corner(self):
        # Perspective trapezoid as clicked: top-left, top-right, bottom-right, bottom-left.
        corners = [[400, 200], [880, 210], [1180, 650], [90, 640]]
        matrix = matrix_from_corners(corners)
        expected = [(0.0, 0.0), (12.0, 0.0), (12.0, 12.0), (0.0, 12.0)]
        for (x, y), field in zip(corners, expected):
            mapped = project_point(matrix, x, y)
            self.assertAlmostEqual(mapped[0], field[0], places=3, msg=f"x of corner {x},{y}")
            self.assertAlmostEqual(mapped[1], field[1], places=3, msg=f"y of corner {x},{y}")
        # A point on the left image edge lies on the field's left edge (x = 0), not the top edge.
        left_mid = project_point(matrix, (400 + 90) / 2, (200 + 640) / 2)
        self.assertAlmostEqual(left_mid[0], 0.0, places=3)
        self.assertTrue(0.0 < left_mid[1] < 12.0)
        self.assertLess(reprojection_error_px(corners), 0.01)

    def test_homography_rejects_bow_tie_corners(self):
        with self.assertRaisesRegex(ValueError, "cross over"):
            matrix_from_corners([[10, 10], [130, 10], [10, 130], [130, 130]])

    def test_homography_rejects_reversed_corner_order(self):
        with self.assertRaisesRegex(ValueError, "reverse order"):
            matrix_from_corners([[10, 10], [10, 130], [130, 130], [130, 10]])

    def test_homography_rejects_three_collinear_corners(self):
        with self.assertRaisesRegex(ValueError, "straight line"):
            matrix_from_corners([[10, 10], [70, 10], [130, 10], [70, 130]])
        with self.assertRaisesRegex(ValueError, "straight line"):
            matrix_from_corners([[10, 10], [130, 10], [130, 130], [131, 250]])

    def test_ocr_consensus_outvotes_single_earlier_high_reading_and_masks_polygon(self):
        result = _consensus([{"t_s": 1, "value": 100, "confidence": .9},
                             {"t_s": 2, "value": 90, "confidence": .9},
                             {"t_s": 3, "value": 90, "confidence": .9}])
        self.assertEqual(result["value"], 90)
        self.assertTrue(result["uncorroborated_prior_high"])
        self.assertEqual(result["outvoted_values"], [100])
        self.assertAlmostEqual(result["agreement"], 2 / 3, places=3)

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
        # The score progressed from 90 to 100 and then held: the stable final reading wins
        # even though the earlier value has more frames overall.
        rows = ([{"t_s": float(i), "value": 90, "confidence": .99} for i in range(8)] +
                [{"t_s": float(i), "value": 100, "confidence": .99} for i in range(8, 12)])
        result = _consensus(rows)
        self.assertEqual(result["value"], 100)
        self.assertEqual(result["agreement"], 1.0)
        self.assertFalse(result["later_lower_reading"])

    def test_ocr_consensus_single_outlier_is_outvoted(self):
        rows = [{"t_s": float(i), "value": 88, "confidence": .95} for i in range(11)]
        for position in (0, 5, 11):
            samples = [dict(row) for row in rows]
            samples.insert(position, {"t_s": 99.0, "value": 188, "confidence": .99})
            result = _consensus(samples)
            self.assertEqual(result["value"], 88, f"outlier at {position}")
            self.assertAlmostEqual(result["agreement"], 11 * .95 / (11 * .95 + .99), places=3)
            self.assertTrue(result["uncorroborated_prior_high"])

    def test_ocr_consensus_outlier_after_score_progression_is_outvoted(self):
        rows = ([{"t_s": float(i), "value": 90, "confidence": .9} for i in range(8)] +
                [{"t_s": 8.0, "value": 100, "confidence": .9}, {"t_s": 9.0, "value": 100, "confidence": .9},
                 {"t_s": 10.0, "value": 188, "confidence": .99}, {"t_s": 11.0, "value": 100, "confidence": .9}])
        self.assertEqual(_consensus(rows)["value"], 100)

    def test_ocr_consensus_weights_votes_by_confidence(self):
        rows = [{"t_s": 0.0, "value": 81, "confidence": .2}, {"t_s": 1.0, "value": 81, "confidence": .2},
                {"t_s": 2.0, "value": 87, "confidence": .95}]
        self.assertEqual(_consensus(rows)["value"], 87)

    def test_ocr_consensus_breaks_ties_with_later_frames(self):
        rows = [{"t_s": 0.0, "value": 76, "confidence": .9}, {"t_s": 1.0, "value": 70, "confidence": .9}]
        self.assertEqual(_consensus(rows)["value"], 70)

    def test_ocr_engine_failure_releases_video(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 300}[prop]
            def release(self): self.released = True

        capture = Capture()
        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=capture), \
             patch.dict(sys.modules, _fake_rapidocr(MagicMock(side_effect=RuntimeError("model load failed")))):
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                read_scoreboard(Path("match.mp4"), ROI, ROI)
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
             patch.dict(sys.modules, _fake_rapidocr(MagicMock())), \
             patch("control_vision.pipeline.scoreboard_ocr._read_digits", side_effect=ocr_values), \
             patch("control_vision.pipeline.scoreboard_ocr.logging.getLogger"):
            result = read_scoreboard(Path("match.mp4"), ROI, ROI, sample_fps=1, tail_seconds=3)
        self.assertEqual(result["frames_sampled"], 2)
        self.assertEqual(result["red"]["value"], 100)
        self.assertTrue(capture.released)

    def test_one_failed_frame_read_does_not_end_ocr_sampling(self):
        class Capture:
            released = False
            position = 0
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 1, cv2.CAP_PROP_FRAME_COUNT: 4}[prop]
            def set(self, _prop, value): self.position = int(value)
            def read(self):
                if self.position == 1:
                    return False, None
                return True, np.zeros((20, 20, 3), dtype=np.uint8)
            def release(self): self.released = True

        capture = Capture()
        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=capture), \
             patch.dict(sys.modules, _fake_rapidocr(MagicMock())), \
             patch("control_vision.pipeline.scoreboard_ocr._read_digits", return_value=(42, .9, "42")), \
             patch("control_vision.pipeline.scoreboard_ocr.logging.getLogger"):
            result = read_scoreboard(Path("match.mp4"), ROI, ROI, sample_fps=1, tail_seconds=4)
        self.assertEqual([row["t_s"] for row in result["red_samples"]], [0.0, 2.0, 3.0])
        self.assertEqual(result["frames_skipped"], 1)
        self.assertEqual(result["red"]["value"], 42)
        self.assertTrue(capture.released)

    def test_ocr_stops_after_many_consecutive_failed_reads(self):
        class Capture:
            reads = 0
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 1, cv2.CAP_PROP_FRAME_COUNT: 100}[prop]
            def set(self, *_): pass
            def read(self):
                self.reads += 1
                return False, None
            def release(self): pass

        capture = Capture()
        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=capture), \
             patch.dict(sys.modules, _fake_rapidocr(MagicMock())), \
             patch("control_vision.pipeline.scoreboard_ocr.logging.getLogger"):
            with self.assertRaisesRegex(RuntimeError, "could not read any frame.*match.mp4"):
                read_scoreboard(Path("match.mp4"), ROI, ROI, sample_fps=1, tail_seconds=100)
        self.assertEqual(capture.reads, 5)

    def test_ocr_failing_on_every_frame_is_an_error_not_an_empty_read(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 1, cv2.CAP_PROP_FRAME_COUNT: 3}[prop]
            def set(self, *_): pass
            def read(self): return True, np.zeros((20, 20, 3), dtype=np.uint8)
            def release(self): self.released = True

        capture = Capture()
        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=capture),              patch.dict(sys.modules, _fake_rapidocr(MagicMock())),              patch("control_vision.pipeline.scoreboard_ocr._read_digits", side_effect=RuntimeError("onnx failed")),              patch("control_vision.pipeline.scoreboard_ocr.logging.getLogger"):
            with self.assertRaisesRegex(RuntimeError, "could not read any frame.*match.mp4"):
                read_scoreboard(Path("match.mp4"), ROI, ROI, sample_fps=1, tail_seconds=3)
        self.assertTrue(capture.released)

    def test_ocr_with_no_digits_still_returns_a_result(self):
        class Capture:
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 1, cv2.CAP_PROP_FRAME_COUNT: 2}[prop]
            def set(self, *_): pass
            def read(self): return True, np.zeros((20, 20, 3), dtype=np.uint8)
            def release(self): pass

        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=Capture()),              patch.dict(sys.modules, _fake_rapidocr(MagicMock())),              patch("control_vision.pipeline.scoreboard_ocr._read_digits", return_value=(None, 0.0, "")):
            result = read_scoreboard(Path("match.mp4"), ROI, ROI, sample_fps=1, tail_seconds=2)
        self.assertEqual(result["frames_sampled"], 2)
        self.assertIsNone(result["red"]["value"])

    def test_invalid_scoreboard_roi_is_rejected_up_front(self):
        capture = MagicMock()
        bad_rois = ([], [[1, 1], [5, 5]], [[1, 1], [1, 1], [1, 1]], [[0, 0], [1, 0], [1, 1], [0, 1]],
                    [[0, 0], [float("nan"), 0], [5, 5]], [["a", 0], [5, 0], [5, 5]])
        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=capture), \
             patch.dict(sys.modules, _fake_rapidocr(MagicMock())):
            for bad in bad_rois:
                with self.assertRaisesRegex(ValueError, "red scoreboard area"):
                    read_scoreboard(Path("match.mp4"), bad, ROI)
            with self.assertRaisesRegex(ValueError, "blue scoreboard area"):
                read_scoreboard(Path("match.mp4"), ROI, [])
        capture.read.assert_not_called()

    def test_scoreboard_roi_outside_frame_is_rejected(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 1, cv2.CAP_PROP_FRAME_COUNT: 3}[prop]
            def set(self, *_): pass
            def read(self): return True, np.zeros((20, 20, 3), dtype=np.uint8)
            def release(self): self.released = True

        capture = Capture()
        outside = [[100, 100], [140, 100], [140, 140], [100, 140]]
        with patch("control_vision.pipeline.scoreboard_ocr.cv2.VideoCapture", return_value=capture), \
             patch.dict(sys.modules, _fake_rapidocr(MagicMock())):
            with self.assertRaisesRegex(ValueError, "blue scoreboard area is outside the video frame"):
                read_scoreboard(Path("match.mp4"), ROI, outside, sample_fps=1, tail_seconds=3)
        self.assertTrue(capture.released)

    def test_missing_ocr_package_names_package_and_version_to_install(self):
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": None}):
            with self.assertRaisesRegex(RuntimeError, r"pip install rapidocr-onnxruntime==1\.4\.4"):
                read_scoreboard(Path("match.mp4"), ROI, ROI)

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

            with patch("control_vision.pipeline.clips.shutil.which", return_value="ffmpeg"), \
                 patch("control_vision.pipeline.clips.subprocess.run", side_effect=run):
                self.assertEqual(create_highlight(Path("match.mp4"), output, 1.0, "ffmpeg"), (0, 4.0))
            self.assertEqual(output.read_bytes(), b"encoded")
            self.assertEqual(len(calls), 2)

    def test_failed_reencode_removes_partial_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "clip.mp4"
            def run(command, **kwargs):
                output.write_bytes(b"partial")
                return SimpleNamespace(returncode=1, stderr="encode failed")

            with patch("control_vision.pipeline.clips.shutil.which", return_value="ffmpeg"), \
                 patch("control_vision.pipeline.clips.subprocess.run", side_effect=run):
                with self.assertRaisesRegex(RuntimeError, "could not create the clip"), \
                     self.assertLogs("control_vision.pipeline.clips", "WARNING"):
                    create_highlight(Path("match.mp4"), output, 5.0, "ffmpeg", duration_s=20.0)
            self.assertFalse(output.exists())

    def test_clip_end_is_clamped_to_video_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "clip.mp4"
            run = MagicMock(side_effect=_ok_run)
            with patch("control_vision.pipeline.clips.shutil.which", return_value="ffmpeg"), \
                 patch("control_vision.pipeline.clips.subprocess.run", run):
                self.assertEqual(create_highlight(Path("match.mp4"), output, 9.0, "ffmpeg", duration_s=10.0),
                                 (6.0, 10.0))
            command = run.call_args_list[0].args[0]
            self.assertEqual(command[command.index("-t") + 1], "4.000")

    def test_clip_duration_is_read_from_video_when_not_given(self):
        with tempfile.TemporaryDirectory() as folder:
            video = _write_video(Path(folder) / "match.avi", frames=60, fps=30.0)
            output = Path(folder) / "clip.mp4"
            with patch("control_vision.pipeline.clips.shutil.which", return_value="ffmpeg"), \
                 patch("control_vision.pipeline.clips.subprocess.run", side_effect=_ok_run):
                start, end = create_highlight(video, output, 1.5, "ffmpeg")
            self.assertEqual(start, 0.0)
            self.assertAlmostEqual(end, 2.0, places=3)

    def test_clip_event_after_video_end_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "clip.mp4"
            run = MagicMock(side_effect=_ok_run)
            with patch("control_vision.pipeline.clips.shutil.which", return_value="ffmpeg"), \
                 patch("control_vision.pipeline.clips.subprocess.run", run):
                with self.assertRaisesRegex(ValueError, "after the end"):
                    create_highlight(Path("match.mp4"), output, 12.0, "ffmpeg", duration_s=10.0)
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_missing_ffmpeg_says_where_to_download_it(self):
        with patch("control_vision.pipeline.clips.shutil.which", return_value=None), \
             patch.dict(sys.modules, {"imageio_ffmpeg": None}):
            with self.assertRaisesRegex(RuntimeError, r"https://ffmpeg\.org/download\.html"):
                create_highlight(Path("match.mp4"), Path("clip.mp4"), 5.0, duration_s=20.0)

    def test_configured_ffmpeg_path_that_does_not_exist_is_named(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = str(Path(folder) / "tools" / "ffmpeg.exe")
            with patch("control_vision.pipeline.clips.subprocess.run") as run:
                with self.assertRaises(RuntimeError) as caught:
                    create_highlight(Path("match.mp4"), Path(folder) / "clip.mp4", 5.0, missing, duration_s=20.0)
            run.assert_not_called()
            self.assertIn(missing, str(caught.exception))
            self.assertIn("does not exist", str(caught.exception))

    def test_hung_ffmpeg_times_out_and_removes_partial_clip(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "clip.mp4"
            def run(command, **kwargs):
                self.assertIn("timeout", kwargs)
                output.write_bytes(b"partial")
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])

            with patch("control_vision.pipeline.clips.shutil.which", return_value="ffmpeg"),                  patch("control_vision.pipeline.clips.subprocess.run", side_effect=run):
                with self.assertRaisesRegex(RuntimeError, "did not finish"):
                    create_highlight(Path("match.mp4"), output, 5.0, "ffmpeg", duration_s=20.0)
            self.assertFalse(output.exists())

    def test_ffmpeg_that_cannot_start_gives_plain_message(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "clip.mp4"
            with patch("control_vision.pipeline.clips.shutil.which", return_value="ffmpeg"), \
                 patch("control_vision.pipeline.clips.subprocess.run", side_effect=FileNotFoundError(2, "missing")):
                with self.assertRaisesRegex(RuntimeError, "could not be started"):
                    create_highlight(Path("match.mp4"), output, 5.0, "ffmpeg", duration_s=20.0)


if __name__ == "__main__":
    unittest.main()
