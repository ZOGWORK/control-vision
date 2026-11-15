import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from control_vision import __main__ as entry  # noqa: E402
from control_vision import smoke_test  # noqa: E402


class BundledWeightTests(unittest.TestCase):
    def test_only_rapidocr_models_are_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = root / "rapidocr_onnxruntime" / "models" / "ch_PP-OCRv4_det_infer.onnx"
            allowed.parent.mkdir(parents=True)
            allowed.write_bytes(b"x")
            (root / "models").mkdir()
            (root / "models" / "yolo11n-biobuzz.onnx").write_bytes(b"x")
            (root / "torch").mkdir()
            (root / "torch" / "stray.PT").write_bytes(b"x")
            (root / "torch" / "notes.txt").write_bytes(b"x")
            self.assertEqual(smoke_test.find_bundled_weights(root), ["models/yolo11n-biobuzz.onnx", "torch/stray.PT"])


class FrozenEntryPointTests(unittest.TestCase):
    def test_frozen_build_refuses_to_act_as_python(self):
        # Ultralytics' auto-installer runs `sys.executable -m pip ...`; the
        # frozen app must exit instead of opening a second window.
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch("control_vision.app.main_window.run") as run_gui:
            self.assertEqual(entry.main(["-m", "pip", "install", "onnx"]), 2)
            self.assertEqual(entry.main(["-c", "print(1)"]), 2)
        run_gui.assert_not_called()

    def test_frozen_build_disables_ultralytics_autoinstall(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YOLO_AUTOINSTALL", None)
            os.environ.pop("YOLO_OFFLINE", None)
            with mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch("control_vision.app.main_window.run") as run_gui:
                self.assertEqual(entry.main([]), 0)
                self.assertEqual(os.environ["YOLO_AUTOINSTALL"], "false")
                self.assertEqual(os.environ["YOLO_OFFLINE"], "true")
            run_gui.assert_called_once()

    def test_source_run_leaves_ultralytics_settings_alone(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YOLO_AUTOINSTALL", None)
            with mock.patch("control_vision.app.main_window.run"):
                self.assertEqual(entry.main([]), 0)
            self.assertNotIn("YOLO_AUTOINSTALL", os.environ)


class SourceSmokeTest(unittest.TestCase):
    def test_source_tree_smoke_test_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            with mock.patch("builtins.print"):
                code = smoke_test.main(["--smoke-test", "--report", str(report_path)])
            report = report_path.read_text(encoding="utf-8")
        self.assertEqual(code, 0, report)
        for step in ("resources", "qt_window", "ocr", "ffmpeg", "project_import", "ultralytics"):
            self.assertIn(f'"{step}"', report)

    def test_failure_is_reported_not_raised(self):
        with mock.patch.object(smoke_test, "check_qt_window", side_effect=RuntimeError("no display")), \
                mock.patch("builtins.print"):
            report = smoke_test.run(smoke_test.parse_args(["--smoke-test"]))
        self.assertFalse(report["ok"])
        self.assertEqual(report["steps"]["qt_window"], {"ok": False, "seconds": mock.ANY,
                                                        "error": "RuntimeError: no display"})
        self.assertNotIn("ocr", report["steps"])

    def test_tracking_needs_model_and_video_together(self):
        with mock.patch.multiple(smoke_test, check_resources=mock.DEFAULT, check_qt_window=mock.DEFAULT,
                                 check_ocr=mock.DEFAULT, check_project_import=mock.DEFAULT,
                                 check_ultralytics=mock.DEFAULT, check_ffmpeg=mock.DEFAULT) as checks:
            checks["check_ffmpeg"].return_value = ({}, Path("clip.mp4"))
            report = smoke_test.run(smoke_test.parse_args(["--smoke-test", "--model", "m.onnx"]))
        self.assertFalse(report["ok"])
        self.assertIn("--model and --video must be given together", report["error"])


if __name__ == "__main__":
    unittest.main()
