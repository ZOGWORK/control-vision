import csv
import tempfile
import unittest
from pathlib import Path

from control_vision.training.split_manifest import validate_split_manifest


class SplitManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []

    def add(self, split, name, match, content):
        image = self.root / "images" / split / f"{name}.jpg"
        label = self.root / "labels" / split / f"{name}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(content)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        self.rows.append({"image": image.relative_to(self.root).as_posix(), "split": split,
                          "source_match": match, "source": "https://example.org/video",
                          "rights": "annotator owns footage"})

    def write_manifest(self):
        with (self.root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image", "split", "source_match", "source", "rights"])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_one_match_per_split_with_provenance_passes(self):
        self.add("train", "q1-a", "event-a:Q1", b"train-a")
        self.add("train", "q1-b", "event-a:Q1", b"train-b")
        self.add("val", "q2-a", "event-b:Q2", b"val")
        self.add("holdout", "q3-a", "event-c:Q3", b"holdout")
        self.write_manifest()
        report = validate_split_manifest(self.root)
        self.assertEqual(report["images"], {"train": 2, "val": 1, "holdout": 1})
        self.assertEqual(report["matches"], 3)
        self.assertEqual(len(report["manifest_sha256"]), 64)

    def test_different_frames_from_one_match_cannot_cross_splits(self):
        self.add("train", "q1-first", "Event-A:Q1", b"first frame")
        self.add("val", "q1-later", "event-a:q1", b"different frame")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "appears in both train and val"):
            validate_split_manifest(self.root)

    def test_byte_identical_image_cannot_cross_splits_even_with_other_match_id(self):
        self.add("train", "first", "event-a:Q1", b"same pixels")
        self.add("val", "other", "event-b:Q9", b"same pixels")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "identical image content"):
            validate_split_manifest(self.root)

    def test_unlisted_image_and_missing_label_fail_closed(self):
        self.add("train", "q1", "event-a:Q1", b"train")
        self.add("val", "q2", "event-b:Q2", b"val")
        self.write_manifest()
        stray = self.root / "images" / "val" / "stray.jpg"
        stray.write_bytes(b"unlisted")
        with self.assertRaisesRegex(ValueError, "absent from the manifest"):
            validate_split_manifest(self.root)
        stray.unlink()
        (self.root / "labels" / "val" / "q2.txt").unlink()
        with self.assertRaisesRegex(ValueError, "missing label"):
            validate_split_manifest(self.root)

    def test_manifest_cannot_reference_parent_path(self):
        self.add("train", "q1", "event-a:Q1", b"train")
        self.add("val", "q2", "event-b:Q2", b"val")
        self.rows[0]["image"] = "images/train/../../private.jpg"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "invalid image path"):
            validate_split_manifest(self.root)


if __name__ == "__main__":
    unittest.main()
