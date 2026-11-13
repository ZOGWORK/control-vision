import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from control_vision.training.train_yolo import publish_model_artifacts, reserve_model_version, validate_dataset


class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
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

    def test_model_version_is_reserved_across_concurrent_runs(self):
        models = self.root / "models"
        with reserve_model_version(models, "1.0.0") as (onnx, metadata):
            self.assertFalse(onnx.exists())
            self.assertFalse(metadata.exists())
            with self.assertRaisesRegex(ValueError, "already being trained"):
                with reserve_model_version(models, "1.0.0"):
                    pass
        self.assertFalse((models / ".yolo11n-biobuzz-v1.0.0.lock").exists())

    def test_existing_model_and_invalid_version_are_rejected(self):
        models = self.root / "models"
        models.mkdir()
        (models / "yolo11n-biobuzz-v1.0.0.onnx").write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            with reserve_model_version(models, "1.0.0"):
                pass
        self.assertFalse((models / ".yolo11n-biobuzz-v1.0.0.lock").exists())
        with self.assertRaisesRegex(ValueError, "Model version"):
            with reserve_model_version(models, "../escape"):
                pass

    def test_partial_artifacts_are_removed_after_publish_failure(self):
        models = self.root / "models"
        exported = self.root / "trained.onnx"
        exported.write_bytes(b"trained model")
        with reserve_model_version(models, "1.0.0") as (onnx, metadata):
            def fail_dump(record, stream, indent):
                stream.write("{partial")
                raise OSError("disk full")

            with patch("control_vision.training.train_yolo.json.dump", side_effect=fail_dump):
                with self.assertRaisesRegex(OSError, "disk full"):
                    publish_model_artifacts(exported, onnx, metadata, {"version": "1.0.0"})
            self.assertFalse(onnx.exists())
            self.assertFalse(metadata.exists())
        with reserve_model_version(models, "1.0.0") as (onnx, metadata):
            publish_model_artifacts(exported, onnx, metadata, {"version": "1.0.0"})
        self.assertEqual(onnx.read_bytes(), b"trained model")


if __name__ == "__main__":
    unittest.main()
