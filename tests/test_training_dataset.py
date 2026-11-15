import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.training.train_yolo import (check_onnx_model, class_names, dataset_hash, model_version_state,
                                                publish_model_artifacts, reserve_model_version,
                                                training_split_provenance, validate_dataset, validation_metrics)
from control_vision.training.extract_frames import extract

REPO_ROOT = Path(__file__).resolve().parents[1]


def complete_record(version="1.0.0"):
    return {"version": version, "date": "2026-11-14", "dataset_sha256": "a" * 64, "classes": ["robot"],
            "training_config": {"epochs": 1}, "validation_metrics": {"overall": {"mAP50": 0.5}},
            "split_provenance": {"mode": "match_audited", "manifest_sha256": "c" * 64, "matches": 2, "events": 1,
                                 "holdout_events": [], "images": {"train": 1, "val": 1, "holdout": 0}}}


class FrameExtractionTests(unittest.TestCase):
    def test_zero_limit_is_rejected_before_opening_video(self):
        with patch("control_vision.training.extract_frames.cv2.VideoCapture") as capture:
            with self.assertRaisesRegex(ValueError, "--max-frames"):
                extract(Path("match.mp4"), Path("unused"), max_frames=0)
            capture.assert_not_called()

    def test_early_decode_failure_reports_video_and_releases_capture(self):
        class TruncatedCapture:
            released = False
            reads = 0

            def __init__(self, valid_frames):
                self.valid_frames = valid_frames

            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 10}[prop]
            def read(self):
                self.reads += 1
                return (self.reads <= self.valid_frames, np.zeros((8, 8, 3), dtype=np.uint8))
            def release(self): self.released = True

        for valid_frames in (2, 9):
            with self.subTest(valid_frames=valid_frames):
                capture = TruncatedCapture(valid_frames)
                with tempfile.TemporaryDirectory() as folder, \
                     patch("control_vision.training.extract_frames.cv2.VideoCapture", return_value=capture):
                    with self.assertRaisesRegex(RuntimeError, "match.mp4"):
                        extract(Path("match.mp4"), Path(folder))
                self.assertTrue(capture.released)


class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
        self.data = self.root / "data.yaml"
        self.config = {"path": str(self.root), "train": "images/train", "val": "images/val"}
        self._sample("train", "one", self._image(30), "0 0.5 0.5 0.3 0.3\n")
        self._sample("val", "two", self._image(90), "0 0.4 0.4 0.2 0.2\n")

    @staticmethod
    def _image(value):
        ok, encoded = cv2.imencode(".jpg", np.full((32, 32, 3), value, dtype=np.uint8))
        assert ok
        return encoded.tobytes()

    def _sample(self, split, name, image_bytes, labels):
        (self.root / "images" / split / f"{name}.jpg").write_bytes(image_bytes)
        (self.root / "labels" / split / f"{name}.txt").write_text(labels, encoding="utf-8")

    def test_valid_data_and_empty_negative_label(self):
        self._sample("val", "three", self._image(120), "0 0.5 0.5 0.2 0.2\n")
        (self.root / "labels" / "val" / "two.txt").write_text("", encoding="utf-8")
        self.assertEqual(validate_dataset(self.data, self.config, 1), self.root)

    def test_portable_yaml_relative_dataset_path(self):
        self.config["path"] = "."
        self.assertEqual(validate_dataset(self.data, self.config, 1), self.root)

    def test_ambiguous_relative_dataset_path_is_rejected(self):
        other = self.root / "other"
        (other / "images" / "train").mkdir(parents=True)
        self.config["path"] = "."
        previous = Path.cwd()
        try:
            os.chdir(other)
            with self.assertRaisesRegex(ValueError, "Ambiguous relative dataset path"):
                validate_dataset(self.data, self.config, 1)
        finally:
            os.chdir(previous)

    def test_absolute_splits_do_not_trigger_relative_path_ambiguity(self):
        self.config.update(path=".", train=str(self.root / "images" / "train"),
                           val=str(self.root / "images" / "val"))
        self.assertEqual(validate_dataset(self.data, self.config, 1), self.root)

    def test_exact_image_leakage_is_rejected(self):
        self._sample("val", "two", self._image(30), "0 0.4 0.4 0.2 0.2\n")
        with self.assertRaisesRegex(ValueError, "train and val"):
            validate_dataset(self.data, self.config, 1)

    def test_unreadable_image_is_rejected(self):
        (self.root / "images" / "val" / "two.jpg").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "Unreadable val image"):
            validate_dataset(self.data, self.config, 1)

    def test_webp_and_bmp_training_images_are_accepted(self):
        for suffix in (".webp", ".bmp"):
            with self.subTest(suffix=suffix):
                (self.root / "images" / "val" / "two.jpg").unlink(missing_ok=True)
                ok, encoded = cv2.imencode(suffix, np.full((32, 32, 3), 90, dtype=np.uint8))
                self.assertTrue(ok)
                (self.root / "images" / "val" / f"two{suffix}").write_bytes(encoded.tobytes())
                self.assertEqual(validate_dataset(self.data, self.config, 1), self.root)
                (self.root / "images" / "val" / f"two{suffix}").unlink()

    def test_missing_label_is_rejected(self):
        (self.root / "labels" / "val" / "two.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing YOLO label"):
            validate_dataset(self.data, self.config, 1)

    def test_nested_images_are_validated_with_matching_nested_labels(self):
        image = self.root / "images" / "train" / "round1" / "third.jpg"
        image.parent.mkdir()
        image.write_bytes(self._image(150))
        with self.assertRaisesRegex(ValueError, "Missing YOLO label"):
            validate_dataset(self.data, self.config, 1)
        label = self.root / "labels" / "train" / "round1" / "third.txt"
        label.parent.mkdir()
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        self.assertEqual(validate_dataset(self.data, self.config, 1), self.root)

    def test_invalid_class_and_out_of_frame_box_are_rejected(self):
        for bad_label in ("1 0.5 0.5 0.3 0.3\n", "0 0.9 0.5 0.4 0.3\n", "0 nan 0.5 0.3 0.3\n"):
            with self.subTest(bad_label=bad_label):
                (self.root / "labels" / "train" / "one.txt").write_text(bad_label, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "out of range"):
                    validate_dataset(self.data, self.config, 1)

    def test_training_requires_robot_labels(self):
        (self.root / "labels" / "train" / "one.txt").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no labeled robots"):
            validate_dataset(self.data, self.config, 1)

    def _source_sample(self, split, folder, name, value):
        (self.root / "images" / split / folder).mkdir(parents=True, exist_ok=True)
        (self.root / "labels" / split / folder).mkdir(parents=True, exist_ok=True)
        self._sample(split, f"{folder}/{name}", self._image(value), "0 0.5 0.5 0.2 0.2\n")

    def test_validation_split_requires_robot_labels(self):
        (self.root / "labels" / "val" / "two.txt").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Validation split has no labeled robots"):
            validate_dataset(self.data, self.config, 1)

    def test_video_folder_in_two_splits_is_rejected(self):
        self._source_sample("train", "Q12-0a1b2c3d4e", "000000060", 140)
        self._source_sample("val", "Q12-0a1b2c3d4e", "000000120", 200)
        with self.assertRaisesRegex(ValueError, "Q12-0a1b2c3d4e.*both images/train and images/val"):
            validate_dataset(self.data, self.config, 1)

    def test_video_folder_in_training_and_holdout_is_rejected(self):
        self._source_sample("train", "Q7-ffff", "000000060", 140)
        (self.root / "images" / "holdout" / "Q7-ffff").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Q7-ffff.*images/holdout"):
            validate_dataset(self.data, self.config, 1)

    def test_holdout_copy_of_training_image_is_rejected(self):
        holdout = self.root / "images" / "holdout"
        holdout.mkdir()
        (holdout / "copy.jpg").write_bytes((self.root / "images" / "train" / "one.jpg").read_bytes())
        with self.assertRaisesRegex(ValueError, "Identical image appears in train and holdout: copy.jpg"):
            validate_dataset(self.data, self.config, 1)
        (holdout / "copy.jpg").write_bytes(self._image(250))
        self.assertEqual(validate_dataset(self.data, self.config, 1), self.root)

    def test_distinct_video_folders_are_accepted(self):
        self._source_sample("train", "Q1-aaaa", "000000060", 140)
        self._source_sample("val", "Q2-bbbb", "000000060", 200)
        self.assertEqual(validate_dataset(self.data, self.config, 1), self.root)

    def _manifest(self, val_match):
        (self.root / "split_manifest.csv").write_text(
            "image,split,source_match,source,rights\n"
            "images/train/one.jpg,train,USWASAM1:Q1,team video,team-owned\n"
            f"images/val/two.jpg,val,{val_match},team video,team-owned\n", encoding="utf-8")

    def test_split_manifest_errors_are_reported_when_present(self):
        self._manifest("uswasam1:q01")
        for exploratory in (False, True):
            with self.subTest(exploratory=exploratory):
                with self.assertRaisesRegex(ValueError, "split_manifest.csv: match .* appears in both train and val"):
                    training_split_provenance(self.root, self.config, exploratory)

    def test_missing_manifest_is_refused_unless_exploratory(self):
        with self.assertRaisesRegex(ValueError, "split_manifest.csv is missing from the dataset root"):
            training_split_provenance(self.root, self.config)
        self.assertEqual(training_split_provenance(self.root, self.config, exploratory=True),
                         {"mode": "exploratory_unverified", "manifest_sha256": None})

    def test_valid_manifest_is_match_audited_even_with_exploratory_flag(self):
        self._manifest("USWASAM1:Q2")
        audit = training_split_provenance(self.root, self.config)
        self.assertEqual(audit, {"mode": "match_audited",
                                 "manifest_sha256": hashlib.sha256((self.root / "split_manifest.csv").read_bytes())
                                 .hexdigest(),
                                 "matches": 2, "events": 1, "holdout_events": [],
                                 "images": {"train": 1, "val": 1, "holdout": 0}})
        self.assertEqual(training_split_provenance(self.root, self.config, exploratory=True), audit)

    def test_manifest_requires_standard_split_directories(self):
        self._manifest("USWASAM1:Q2")
        (self.root / "images" / "extra").mkdir()
        for split in ("train", "val"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, f"{split} in data.yaml must be images/{split}"):
                    training_split_provenance(self.root, {**self.config, split: "images/extra"})

    def test_trained_image_type_outside_manifest_audit_is_refused(self):
        self._manifest("USWASAM1:Q2")
        (self.root / "images" / "train" / "extra.avif").write_bytes(b"not audited")
        with self.assertRaisesRegex(ValueError, r"images/train has 2 image\(s\) but the manifest lists 1"):
            training_split_provenance(self.root, self.config)

    def test_dataset_hash_covers_split_files_and_yaml_only(self):
        self.data.write_text("names: [robot]\n", encoding="utf-8")
        original = dataset_hash(self.root, self.config, self.data)
        unlabeled = self.root / "images" / "unlabeled" / "match-1"
        unlabeled.mkdir(parents=True)
        (unlabeled / "000000000.jpg").write_bytes(self._image(5))
        self.assertEqual(dataset_hash(self.root, self.config, self.data), original)
        (self.root / "labels" / "val" / "two.txt").write_text("0 0.4 0.4 0.3 0.3\n", encoding="utf-8")
        changed_label = dataset_hash(self.root, self.config, self.data)
        self.assertNotEqual(changed_label, original)
        self.data.write_text("names: [robot, pollen, nectar]\n", encoding="utf-8")
        self.assertNotEqual(dataset_hash(self.root, self.config, self.data), changed_label)

    def test_dataset_hash_separates_file_names_from_contents(self):
        labels = self.root / "labels" / "train"
        (labels / "a.txt").write_text("x", encoding="utf-8")
        (labels / "b.txt").write_text("y", encoding="utf-8")
        self.data.write_text("{}", encoding="utf-8")
        two_files = dataset_hash(self.root, self.config, self.data)
        # Without a separator this one file hashes the same bytes as a.txt + b.txt.
        (labels / "b.txt").unlink()
        (labels / "a.txt").write_text("xlabels/train/b.txty", encoding="utf-8")
        self.assertNotEqual(dataset_hash(self.root, self.config, self.data), two_files)


class ClassNameTests(unittest.TestCase):
    def test_supported_class_maps(self):
        self.assertEqual(class_names({"names": {0: "robot"}}), ["robot"])
        self.assertEqual(class_names({"names": ["robot", "pollen", "nectar"]}), ["robot", "pollen", "nectar"])

    def test_broken_class_maps_raise_plain_value_errors(self):
        cases = {"class IDs 0, 1, 2": {"names": {1: "robot"}},
                 "must be a mapping": None,
                 "needs a names list": {"path": "."},
                 "Supported classes": {"names": ["pollen", "robot"]}}
        for message, config in cases.items():
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    class_names(config)


class ModelPublishingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.models = self.root / "models"
        self.exported = self.root / "trained.onnx"
        self.exported.write_bytes(b"trained model bytes")

    def _publish(self, record=None):
        with reserve_model_version(self.models, "1.0.0") as (onnx, metadata):
            return onnx, metadata, publish_model_artifacts(self.exported, onnx, metadata, record or complete_record())

    def test_model_version_is_reserved_across_concurrent_runs(self):
        lock = self.models / ".yolo11n-biobuzz-v1.0.0.lock"
        with reserve_model_version(self.models, "1.0.0") as (onnx, metadata):
            self.assertFalse(onnx.exists())
            self.assertFalse(metadata.exists())
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["pid"], os.getpid())
            with self.assertRaisesRegex(ValueError, f"is locked by .*lock \\(PID {os.getpid()}, started "):
                with reserve_model_version(self.models, "1.0.0"):
                    pass
        self.assertFalse(lock.exists())

    def test_version_must_be_three_numbers(self):
        for version in ("1.0", "1..0", "banana", "1.0.0-rc1", "../escape", "v1.0.0", "1.0.0 "):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "three numbers"):
                    with reserve_model_version(self.models, version):
                        pass
        with reserve_model_version(self.models, "10.2.33"):
            pass

    def test_published_version_is_never_overwritten(self):
        onnx, metadata, record = self._publish()
        self.assertEqual(onnx.read_bytes(), b"trained model bytes")
        written = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(written, record)
        self.assertEqual(written["onnx"], onnx.name)
        self.assertEqual(len(written["onnx_sha256"]), 64)
        self.assertEqual(model_version_state(self.models, "1.0.0")[0], "published")
        self.exported.write_bytes(b"other model")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self._publish()
        self.assertEqual(onnx.read_bytes(), b"trained model bytes")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite existing model file"):
            publish_model_artifacts(self.exported, onnx, metadata, complete_record())
        self.assertEqual(onnx.read_bytes(), b"trained model bytes")
        self.assertEqual(json.loads(metadata.read_text(encoding="utf-8")), record)

    def test_onnx_without_matching_record_is_incomplete(self):
        self.models.mkdir()
        onnx = self.models / "yolo11n-biobuzz-v1.0.0.onnx"
        onnx.write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "incomplete artifact.*JSON record is missing.*delete these files"):
            with reserve_model_version(self.models, "1.0.0"):
                pass
        self.assertFalse((self.models / ".yolo11n-biobuzz-v1.0.0.lock").exists())
        onnx.unlink()
        onnx, metadata, _ = self._publish()
        onnx.write_bytes(b"trained model bytez")
        state, message = model_version_state(self.models, "1.0.0")
        self.assertEqual(state, "incomplete")
        self.assertIn("does not match the SHA-256", message)

    def test_record_is_published_last_and_failures_leave_nothing(self):
        real_link = os.link
        order = []

        def failing_second_link(source, target):
            order.append(Path(target).suffix)
            if len(order) == 2:
                raise OSError("disk full")
            real_link(source, target)

        with patch("control_vision.training.train_yolo.os.link", side_effect=failing_second_link):
            with self.assertRaisesRegex(OSError, "disk full"):
                self._publish()
        self.assertEqual(order, [".onnx", ".json"])
        self.assertEqual(sorted(p.name for p in self.models.iterdir()), [])
        onnx, _, _ = self._publish()
        self.assertEqual(onnx.read_bytes(), b"trained model bytes")

    def test_blank_record_fields_are_refused(self):
        for field in ("dataset_sha256", "validation_metrics", "training_config"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"without {field}"):
                    self._publish({**complete_record(), field: {} if field != "dataset_sha256" else ""})
                self.assertEqual(model_version_state(self.models, "1.0.0")[0], "absent")
        with self.assertRaisesRegex(ValueError, "SHA-256 dataset hash"):
            self._publish({**complete_record(), "dataset_sha256": "unknown"})

    def test_record_without_valid_split_provenance_is_refused(self):
        record = complete_record()
        audited = record["split_provenance"]
        cases = {"missing": {key: value for key, value in record.items() if key != "split_provenance"},
                 "unknown mode": {**record, "split_provenance": {"mode": "trust_me", "manifest_sha256": None}},
                 "audited without hash": {**record, "split_provenance": {**audited, "manifest_sha256": None}},
                 "audited without counts": {**record, "split_provenance": {"mode": "match_audited",
                                                                           "manifest_sha256": "c" * 64}},
                 "exploratory with hash": {**record, "split_provenance": {"mode": "exploratory_unverified",
                                                                          "manifest_sha256": "c" * 64}}}
        for name, bad in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "split_provenance"):
                    self._publish(bad)
                self.assertEqual(model_version_state(self.models, "1.0.0")[0], "absent")
        _, metadata, _ = self._publish({**record, "split_provenance": {"mode": "exploratory_unverified",
                                                                      "manifest_sha256": None}})
        self.assertEqual(json.loads(metadata.read_text(encoding="utf-8"))["split_provenance"]["mode"],
                         "exploratory_unverified")

    def _interrupt(self, mode):
        script = textwrap.dedent("""
            import json, os, sys
            from pathlib import Path
            import control_vision.training.train_yolo as ty
            models, exported, mode, record = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
            real_write, real_link = ty._write_partial, os.link
            def write_half(final, write, partials):
                def half(stream):
                    stream.write(exported.read_bytes()[:4]); stream.flush(); os._exit(9)
                real_write(final, half, partials)
            def link_then_die(source, target):
                real_link(source, target); os._exit(9)
            if mode == "copy":
                ty._write_partial = write_half
            else:
                ty.os.link = link_then_die
            with ty.reserve_model_version(models, "1.0.0") as (onnx, metadata):
                ty.publish_model_artifacts(exported, onnx, metadata, record)
        """)
        child = subprocess.run([sys.executable, "-c", script, str(self.models), str(self.exported), mode,
                                json.dumps(complete_record())], cwd=REPO_ROOT, capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, timeout=120)
        self.assertEqual(child.returncode, 9, child.stderr)
        return self.models / ".yolo11n-biobuzz-v1.0.0.lock"

    def test_killed_during_copy_leaves_no_final_file_and_a_reported_lock(self):
        lock = self._interrupt("copy")
        self.assertFalse((self.models / "yolo11n-biobuzz-v1.0.0.onnx").exists())
        owner = json.loads(lock.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError,
                                    f"locked by .*lock \\(PID {owner['pid']}, started .*delete the lock file"):
            with reserve_model_version(self.models, "1.0.0"):
                pass
        self.assertTrue(lock.exists())
        lock.unlink()
        onnx, _, _ = self._publish()
        self.assertEqual(onnx.read_bytes(), b"trained model bytes")

    def test_killed_between_onnx_and_record_is_incomplete_not_published(self):
        lock = self._interrupt("link")
        self.assertTrue((self.models / "yolo11n-biobuzz-v1.0.0.onnx").exists())
        self.assertFalse((self.models / "yolo11n-biobuzz-v1.0.0.json").exists())
        lock.unlink()
        self.assertEqual(model_version_state(self.models, "1.0.0")[0], "incomplete")
        with self.assertRaisesRegex(ValueError, "incomplete artifact and was never published"):
            self._publish()


def fake_val_result(class_ids=(0,), ap50=(0.8,), ap=(0.5,), summary=None):
    box = types.SimpleNamespace(ap50=list(ap50), ap=list(ap), map50=0.8, map=0.5, ap_class_index=list(class_ids))
    return types.SimpleNamespace(box=box, ap_class_index=list(class_ids),
                                 results_dict={"metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.5}
                                 if summary is None else summary)


class ValidationMetricTests(unittest.TestCase):
    def test_per_class_metrics_are_recorded(self):
        metrics = validation_metrics(fake_val_result((0, 2), (0.9, 0.7), (0.6, 0.4)), ["robot", "pollen", "nectar"])
        self.assertEqual(metrics["per_class"], {"robot": {"mAP50": 0.9, "mAP50-95": 0.6}, "pollen": None,
                                                "nectar": {"mAP50": 0.7, "mAP50-95": 0.4}})
        self.assertEqual(metrics["overall"], {"mAP50": 0.8, "mAP50-95": 0.5})
        self.assertEqual(metrics["split"], "val")

    def test_missing_metrics_are_refused(self):
        for results in (None, types.SimpleNamespace(results_dict={}), fake_val_result(summary={}),
                        fake_val_result((1,)), fake_val_result(ap50=(float("nan"),))):
            with self.subTest(results=results):
                with self.assertRaisesRegex(ValueError, "not published"):
                    validation_metrics(results, ["robot", "pollen", "nectar"])


class FakeOnnxRuntime:
    def __init__(self, shape=(1, 5, 21), names="{0: 'robot'}", error=None):
        self.shape, self.names, self.error, self.feeds = shape, names, error, []
        runtime = self

        class Session:
            def __init__(self, path, providers):
                self.path = path

            def get_inputs(self):
                return [types.SimpleNamespace(name="images")]

            def run(self, output_names, feeds):
                runtime.feeds.append(feeds)
                if runtime.error:
                    raise RuntimeError(runtime.error)
                return [np.zeros(runtime.shape, dtype=np.float32)]

            def get_modelmeta(self):
                return types.SimpleNamespace(custom_metadata_map={"names": runtime.names})

        self.InferenceSession = Session


def fake_onnx(opset=12):
    return types.SimpleNamespace(load=lambda path: types.SimpleNamespace(
        opset_import=[types.SimpleNamespace(domain="", version=opset),
                      types.SimpleNamespace(domain="ai.onnx.ml", version=99)]))


class OnnxCheckTests(unittest.TestCase):
    def test_runs_inference_and_checks_output_and_names(self):
        runtime = FakeOnnxRuntime()
        check = check_onnx_model(Path("m.onnx"), ["robot"], 64, fake_onnx(17), runtime)
        self.assertEqual(check["opset"], 17)
        self.assertEqual(check["output_shape"], [1, 5, 21])
        self.assertEqual(runtime.feeds[0]["images"].shape, (1, 3, 64, 64))

    def test_bad_exports_are_refused(self):
        cases = [("opset 11", fake_onnx(11), FakeOnnxRuntime()),
                 ("does not match 1 x 5", fake_onnx(), FakeOnnxRuntime(shape=(1, 7, 21))),
                 ("class names", fake_onnx(), FakeOnnxRuntime(names="{0: 'pollen'}")),
                 ("class names", fake_onnx(), FakeOnnxRuntime(names=None)),
                 ("smoke check failed: boom", fake_onnx(), FakeOnnxRuntime(error="boom"))]
        for message, onnx, runtime in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    check_onnx_model(Path("m.onnx"), ["robot"], 64, onnx, runtime)


if __name__ == "__main__":
    unittest.main()
