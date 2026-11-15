"""End-to-end checks of train_yolo.main() with fake Ultralytics, ONNX, and ONNX Runtime modules.

No training packages are needed: every import the script makes is replaced
in sys.modules, so these tests run with requirements-extract.txt only.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.training import train_yolo
from control_vision.training.train_yolo import dataset_hash, publish_model_artifacts, reserve_model_version


def _int_keys(mapping):
    return {int(key) if key.isdigit() else key: value for key, value in mapping.items()}


class FakeTraining:
    """Records what main() asks of Ultralytics and ONNX Runtime."""

    def __init__(self, work: Path):
        self.work = work
        self.calls: list[tuple[str, dict]] = []
        self.feeds: list[dict] = []
        self.env_at_init: list[tuple] = []
        self.opset = 12
        self.metadata_names = "{0: 'robot'}"
        self.run_error = None
        self.output_channels = 5
        self.val_result = types.SimpleNamespace(
            results_dict={"metrics/mAP50(B)": 0.81, "metrics/mAP50-95(B)": 0.52},
            ap_class_index=[0],
            box=types.SimpleNamespace(ap50=[0.81], ap=[0.52], map50=0.81, map=0.52, ap_class_index=[0]))
        self.on_train = None

    def modules(self):
        fake = self

        class YOLO:
            def __init__(self, weights):
                self.weights = Path(weights)
                fake.calls.append(("init", {"weights": weights}))
                fake.env_at_init.append((os.environ.get("YOLO_AUTOINSTALL"), os.environ.get("YOLO_OFFLINE")))

            def train(self, **kwargs):
                fake.calls.append(("train", kwargs))
                weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
                weights.mkdir(parents=True, exist_ok=True)
                (weights / "best.pt").write_bytes(b"best")
                self.trainer = types.SimpleNamespace(best=str(weights / "best.pt"), save_dir=str(weights.parent))
                if fake.on_train:
                    fake.on_train()

            def val(self, **kwargs):
                fake.calls.append(("val", kwargs))
                return fake.val_result

            def export(self, **kwargs):
                fake.calls.append(("export", kwargs))
                exported = self.weights.with_suffix(".onnx")
                exported.write_bytes(b"exported onnx bytes")
                return str(exported)

        class Session:
            def __init__(self, path, providers):
                fake.calls.append(("session", {"path": path, "providers": providers}))

            def get_inputs(self):
                return [types.SimpleNamespace(name="images")]

            def run(self, output_names, feeds):
                fake.feeds.append(feeds)
                if fake.run_error:
                    raise RuntimeError(fake.run_error)
                return [np.zeros((1, fake.output_channels, 8), dtype=np.float32)]

            def get_modelmeta(self):
                return types.SimpleNamespace(custom_metadata_map={"names": fake.metadata_names})

        yaml = types.ModuleType("yaml")
        yaml.YAMLError = type("YAMLError", (Exception,), {})
        yaml.safe_load = lambda text: json.loads(text, object_hook=_int_keys) if text.strip() else None
        yaml.safe_dump = lambda data: json.dumps(data)
        ultralytics = types.ModuleType("ultralytics")
        ultralytics.YOLO = YOLO
        ultralytics.__version__ = "8.4.210"
        onnx = types.ModuleType("onnx")
        onnx.load = lambda path: types.SimpleNamespace(opset_import=[types.SimpleNamespace(domain="", version=fake.opset)])
        onnxruntime = types.ModuleType("onnxruntime")
        onnxruntime.InferenceSession = Session
        return {"yaml": yaml, "ultralytics": ultralytics, "onnx": onnx, "onnxruntime": onnxruntime}

    def called(self, name):
        return [kwargs for call, kwargs in self.calls if call == name]


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.root = self.work / "dataset"
        for split, value in (("train", 30), ("val", 90)):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            ok, encoded = cv2.imencode(".jpg", np.full((32, 32, 3), value, dtype=np.uint8))
            self.assertTrue(ok)
            (self.root / "images" / split / "a.jpg").write_bytes(encoded.tobytes())
            (self.root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        self.manifest = self.root / "split_manifest.csv"
        self.manifest.write_text("image,split,source_match,source,rights\n"
                                 "images/train/a.jpg,train,USWASAM1:Q1,team video,team-owned\n"
                                 "images/val/a.jpg,val,USWASAM1:Q2,team video,team-owned\n", encoding="utf-8")
        self.data = self.root / "data.yaml"
        self.config = {"path": str(self.root), "train": "images/train", "val": "images/val", "names": {"0": "robot"}}
        self._write_config()
        self.base = self.work / "base.pt"
        self.base.write_bytes(b"base")
        self.models = self.work / "models"
        self.runs = self.work / "runs"
        self.fake = FakeTraining(self.work)

    def _write_config(self, text=None):
        self.data.write_text(json.dumps(self.config) if text is None else text, encoding="utf-8")

    def run_main(self, *extra, modules=None):
        argv = ["--data", str(self.data), "--base-weights", str(self.base), "--models-dir", str(self.models),
                "--runs-dir", str(self.runs), *extra]
        stderr, self.stdout = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, modules or self.fake.modules()), patch.dict(os.environ, {}), \
                contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(self.stdout):
            try:
                train_yolo.main(argv)
            except SystemExit as exc:
                return exc.code, stderr.getvalue()
        return 0, stderr.getvalue()

    def record(self):
        return json.loads((self.models / "yolo11n-biobuzz-v1.0.0.json").read_text(encoding="utf-8"))

    def model_files(self):
        return sorted(path.name for path in self.models.iterdir()) if self.models.exists() else []

    def assert_not_trained(self):
        self.assertEqual(self.fake.called("train"), [])
        self.assertEqual([name for name in self.model_files() if not name.startswith(".")], [])

    def test_successful_run_publishes_a_complete_record(self):
        expected_hash = dataset_hash(self.root, json.loads(self.data.read_text()), self.data)
        # A file that changes during training must not change the recorded hash.
        self.fake.on_train = lambda: (self.root / "labels" / "train" / "a.txt").write_text(
            "0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
        code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.model_files(), ["yolo11n-biobuzz-v1.0.0.json", "yolo11n-biobuzz-v1.0.0.onnx"])
        record = json.loads((self.models / "yolo11n-biobuzz-v1.0.0.json").read_text(encoding="utf-8"))
        onnx = self.models / "yolo11n-biobuzz-v1.0.0.onnx"
        self.assertEqual(onnx.read_bytes(), b"exported onnx bytes")
        self.assertEqual(record["onnx_sha256"], train_yolo._sha256(onnx))
        self.assertEqual(record["dataset_sha256"], expected_hash)
        self.assertEqual(record["classes"], ["robot"])
        self.assertEqual(record["validation_metrics"]["per_class"], {"robot": {"mAP50": 0.81, "mAP50-95": 0.52}})
        self.assertEqual(record["validation_metrics"]["overall"], {"mAP50": 0.81, "mAP50-95": 0.52})
        self.assertEqual(record["onnxruntime_check"]["smoke_run"], "passed")
        self.assertGreaterEqual(record["onnxruntime_check"]["opset"], 12)
        self.assertEqual(record["split_provenance"],
                         {"mode": "match_audited", "manifest_sha256": train_yolo._sha256(self.manifest),
                          "matches": 2, "events": 1, "holdout_events": [],
                          "images": {"train": 1, "val": 1, "holdout": 0}})
        self.assertNotIn("UNVERIFIED", self.stdout.getvalue())

        [train] = self.fake.called("train")
        self.assertEqual({key: train[key] for key in ("hsv_h", "hsv_s", "hsv_v", "imgsz", "epochs", "batch",
                                                      "lr0", "patience", "plots")},
                         {"hsv_h": 0.0, "hsv_s": 0.2, "hsv_v": 0.2, "imgsz": 640, "epochs": 100, "batch": 16,
                          "lr0": 0.01, "patience": 20, "plots": False})
        self.assertEqual(train["project"], str(self.runs.resolve()))
        self.assertTrue(Path(train["project"]).is_absolute())
        config = {key: value for key, value in train.items() if key not in ("data", "project", "name")}
        self.assertEqual(record["training_config"], {"model": "yolo11n", **config})
        [val] = self.fake.called("val")
        self.assertEqual(val["split"], "val")
        [export] = self.fake.called("export")
        self.assertEqual(export["format"], "onnx")
        self.assertGreaterEqual(export["opset"], 12)
        self.assertEqual(self.fake.feeds[0]["images"].shape, (1, 3, 640, 640))
        self.assertEqual(set(self.fake.env_at_init), {("False", "True")})

    def test_missing_manifest_is_refused_before_training(self):
        self.manifest.unlink()
        with patch.object(train_yolo, "validate_dataset", side_effect=AssertionError("images checked")):
            code, stderr = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("split_manifest.csv is missing from the dataset root", stderr)
        self.assert_not_trained()
        self.assertEqual(self.fake.called("init"), [])

    def test_exploratory_flag_publishes_a_visibly_unverified_record(self):
        self.manifest.unlink()
        code, stderr = self.run_main("--exploratory-without-manifest")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.record()["split_provenance"], {"mode": "exploratory_unverified", "manifest_sha256": None})
        output = self.stdout.getvalue()
        self.assertIn("UNVERIFIED EXPLORATORY RUN", output)
        self.assertIn("UNVERIFIED: version 1.0.0 is exploratory_unverified", output)

    def test_exploratory_flag_does_not_bypass_an_invalid_manifest(self):
        self.manifest.write_text(self.manifest.read_text(encoding="utf-8").replace("USWASAM1:Q2", "uswasam1:q01"),
                                 encoding="utf-8")
        code, stderr = self.run_main("--exploratory-without-manifest")
        self.assertEqual(code, 2)
        self.assertIn("split_manifest.csv: match", stderr)
        self.assert_not_trained()

    def test_exploratory_flag_with_valid_manifest_is_match_audited(self):
        code, stderr = self.run_main("--exploratory-without-manifest")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.record()["split_provenance"]["mode"], "match_audited")
        self.assertNotIn("UNVERIFIED", self.stdout.getvalue())

    def test_training_folders_outside_the_manifest_layout_are_refused(self):
        (self.root / "images" / "train2").mkdir()
        (self.root / "images" / "train2" / "b.jpg").write_bytes((self.root / "images" / "train" / "a.jpg").read_bytes())
        self.config["train"] = "images/train2"
        self._write_config()
        code, stderr = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("train in data.yaml must be images/train", stderr)
        self.assert_not_trained()

    def test_unsupported_class_map_is_refused_before_training(self):
        for names, message in (({"0": "pollen", "1": "robot"}, "Supported classes"),
                               ({"1": "robot"}, "class IDs 0, 1, 2")):
            with self.subTest(names=names):
                self.config["names"] = names
                self._write_config()
                code, stderr = self.run_main()
                self.assertEqual(code, 2)
                self.assertIn(message, stderr)
                self.assert_not_trained()

    def test_empty_data_yaml_is_a_plain_error(self):
        self._write_config("")
        code, stderr = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("data.yaml must be a mapping", stderr)
        self.assert_not_trained()

    def assert_not_published(self, code, message):
        # Failures after training are not usage errors: plain message, exit status 1.
        self.assertIsInstance(code, str)
        self.assertIn(message, code)
        self.assertEqual(self.model_files(), [])

    def test_old_opset_is_not_published(self):
        self.fake.opset = 11
        code, _ = self.run_main()
        self.assert_not_published(code, "Model version 1.0.0 was not published: Exported ONNX uses opset 11")
        self.assertEqual(self.model_files(), [])

    def test_onnx_runtime_failure_is_not_published(self):
        self.fake.run_error = "bad graph"
        code, _ = self.run_main()
        self.assert_not_published(code, "ONNX Runtime smoke check failed: bad graph")

    def test_wrong_output_shape_or_metadata_is_not_published(self):
        for attribute, value, message in (("output_channels", 7, "does not match 1 x 5"),
                                          ("metadata_names", "{0: 'nectar'}", "class names")):
            with self.subTest(attribute=attribute):
                fake = FakeTraining(self.work)
                setattr(fake, attribute, value)
                self.fake = fake
                code, _ = self.run_main()
                self.assert_not_published(code, message)

    def test_missing_validation_metrics_are_not_published(self):
        self.fake.val_result = None
        code, _ = self.run_main()
        self.assert_not_published(code, "Validation returned no metrics")
        self.assertEqual(self.fake.called("export"), [])

    def test_errors_raised_during_training_keep_their_traceback(self):
        def bad_device():
            raise ValueError("Invalid CUDA 'device=7' requested")

        self.fake.on_train = bad_device
        with self.assertRaisesRegex(ValueError, "Invalid CUDA"):
            self.run_main()
        self.assertEqual(self.model_files(), [])

    def test_existing_version_is_never_overwritten(self):
        exported = self.work / "old.onnx"
        exported.write_bytes(b"released weights")
        with reserve_model_version(self.models, "1.0.0") as (onnx, metadata):
            publish_model_artifacts(exported, onnx, metadata, {
                "version": "1.0.0", "date": "2026-11-01", "dataset_sha256": "b" * 64, "classes": ["robot"],
                "training_config": {"epochs": 1}, "validation_metrics": {"overall": {"mAP50": 0.1}},
                "split_provenance": {"mode": "exploratory_unverified", "manifest_sha256": None}})
        before = {path.name: path.read_bytes() for path in self.models.iterdir()}
        code, stderr = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("Refusing to overwrite existing model version 1.0.0", stderr)
        self.assertEqual(self.fake.called("init"), [])
        self.assertEqual({path.name: path.read_bytes() for path in self.models.iterdir()}, before)

    def test_split_outside_dataset_root_is_refused(self):
        outside = self.work / "elsewhere" / "images" / "val"
        outside.mkdir(parents=True)
        (outside / "a.jpg").write_bytes((self.root / "images" / "val" / "a.jpg").read_bytes())
        self.config["val"] = str(outside)
        self._write_config()
        code, stderr = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("outside images/", stderr)
        self.assert_not_trained()

    def test_invalid_version_is_refused(self):
        code, stderr = self.run_main("--version", "1.0")
        self.assertEqual(code, 2)
        self.assertIn("three numbers", stderr)
        self.assert_not_trained()

    def test_missing_training_package_is_reported_before_training(self):
        for missing in ("onnx", "onnxruntime", "ultralytics", "yaml"):
            with self.subTest(missing=missing):
                modules = self.fake.modules()
                modules[missing] = None  # Makes `import <missing>` raise ImportError.
                code, _ = self.run_main(modules=modules)
                package = "PyYAML" if missing == "yaml" else missing
                self.assertEqual(code, f"The {package} package is missing; install the training packages with "
                                       f"python -m pip install -r requirements-train.txt")
                self.assert_not_trained()
                self.assertEqual(self.fake.called("init"), [])

    def test_network_installs_are_disabled_before_ultralytics_is_used(self):
        with patch.dict(os.environ, {"YOLO_AUTOINSTALL": "True", "YOLO_OFFLINE": "False"}):
            with patch.dict(sys.modules, self.fake.modules()):
                train_yolo.check_training_packages()
            self.assertEqual(os.environ["YOLO_AUTOINSTALL"], "False")
            self.assertEqual(os.environ["YOLO_OFFLINE"], "True")


if __name__ == "__main__":
    unittest.main()
