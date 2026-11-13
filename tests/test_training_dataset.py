import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from control_vision.training.train_yolo import validate_dataset


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

    def test_exact_image_leakage_is_rejected(self):
        self._sample("val", "two", self._image(30), "0 0.4 0.4 0.2 0.2\n")
        with self.assertRaisesRegex(ValueError, "train and val"):
            validate_dataset(self.data, self.config, 1)

    def test_unreadable_image_is_rejected(self):
        (self.root / "images" / "val" / "two.jpg").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "Unreadable val image"):
            validate_dataset(self.data, self.config, 1)

    def test_missing_label_is_rejected(self):
        (self.root / "labels" / "val" / "two.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing YOLO label"):
            validate_dataset(self.data, self.config, 1)

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


if __name__ == "__main__":
    unittest.main()
