import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from control_vision.training.split_manifest import main, validate_split_manifest


FIELDNAMES = ["image", "split", "source_match", "source", "rights"]


class SplitManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []

    def add(self, split, name, match, content, suffix=".jpg"):
        image = self.root / "images" / split / f"{name}{suffix}"
        label = self.root / "labels" / split / f"{name}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(content)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        self.rows.append({"image": image.relative_to(self.root).as_posix(), "split": split,
                          "source_match": match, "source": "https://example.org/video",
                          "rights": "annotator owns footage"})

    def add_baseline(self):
        self.add("train", "q1", "event-a:Q1", b"train")
        self.add("val", "q2", "event-b:Q2", b"val")

    def write_manifest(self, path=None):
        with (path or self.root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_raw_manifest(self, lines):
        (self.root / "split_manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            validate_split_manifest(self.root)

    def test_one_match_per_split_with_provenance_passes(self):
        self.add("train", "q1-a", "event-a:Q1", b"train-a")
        self.add("train", "q1-b", "event-a:Q1", b"train-b")
        self.add("val", "q2-a", "event-b:Q2", b"val")
        self.add("holdout", "q3-a", "event-c:Q3", b"holdout")
        self.write_manifest()
        report = validate_split_manifest(self.root)
        self.assertEqual(report["images"], {"train": 2, "val": 1, "holdout": 1})
        self.assertEqual(report["matches"], 3)
        self.assertEqual(report["events"], 3)
        self.assertEqual(report["holdout_events"], ["event-c"])
        self.assertEqual(len(report["manifest_sha256"]), 64)

    def test_different_frames_from_one_match_cannot_cross_splits(self):
        self.add("train", "q1-first", "Event-A:Q1", b"first frame")
        self.add("val", "q1-later", "event-a:q1", b"different frame")
        self.write_manifest()
        self.assert_rejected("appears in both train and val")

    def test_match_id_spacing_and_leading_zeros_are_normalised(self):
        self.add("train", "q3-first", "event-a : Q03", b"first frame")
        self.add("val", "q3-later", "Event-A:q3", b"different frame")
        self.write_manifest()
        self.assert_rejected("appears in both train and val")

    def test_train_and_val_may_share_an_event_with_different_matches(self):
        self.add("train", "q1", "event-a:Q1", b"train")
        self.add("val", "q2", "event-a:Q2", b"val")
        self.write_manifest()
        report = validate_split_manifest(self.root)
        self.assertEqual(report["matches"], 2)
        self.assertEqual(report["events"], 1)

    def test_holdout_event_cannot_supply_train_frames_from_other_matches(self):
        self.add("train", "stl-q4", "USSTLNorth:Q4", b"train")
        self.add("val", "q2", "event-b:Q2", b"val")
        self.add("holdout", "stl-q3", "USSTLNorth:Q3", b"holdout")
        self.write_manifest()
        self.assert_rejected("holdout event 'usstlnorth' also appears in train")

    def test_holdout_event_cannot_supply_val_frames_from_other_matches(self):
        self.add("train", "q1", "event-a:Q1", b"train")
        self.add("val", "stl-q4", "usstlnorth :Q4", b"val")
        self.add("holdout", "stl-q3", "USSTLNorth:Q3", b"holdout")
        self.write_manifest()
        self.assert_rejected("holdout event 'usstlnorth' also appears in val")

    def test_source_match_without_event_is_rejected(self):
        self.add_baseline()
        for value in ("Q1", " :Q1", "event-a: "):
            with self.subTest(value=value):
                self.rows[0]["source_match"] = value
                self.write_manifest()
                self.assert_rejected("row 2: source_match must look like EVENT:MATCH")

    def test_byte_identical_image_cannot_cross_splits_even_with_other_match_id(self):
        self.add("train", "first", "event-a:Q1", b"same pixels")
        self.add("val", "other", "event-b:Q9", b"same pixels")
        self.write_manifest()
        self.assert_rejected("identical image content")

    def test_unlisted_image_and_missing_label_fail_closed(self):
        self.add_baseline()
        self.write_manifest()
        stray = self.root / "images" / "val" / "stray.jpg"
        stray.write_bytes(b"unlisted")
        self.assert_rejected("absent from the manifest")
        stray.unlink()
        (self.root / "labels" / "val" / "q2.txt").unlink()
        self.assert_rejected("missing label")

    def test_image_folder_outside_known_splits_is_rejected(self):
        self.add_baseline()
        self.write_manifest()
        extra = self.root / "images" / "test" / "extra.jpg"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"other folder")
        self.assert_rejected("absent from the manifest")

    def test_orphan_label_file_is_rejected(self):
        self.add_baseline()
        self.write_manifest()
        (self.root / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        self.assert_rejected("label file\\(s\\) have no manifest image")

    def test_manifest_cannot_reference_parent_path(self):
        self.add_baseline()
        self.rows[0]["image"] = "images/train/../../private.jpg"
        self.write_manifest()
        self.assert_rejected("invalid image path")

    def test_blank_source_is_rejected(self):
        self.add_baseline()
        self.rows[0]["source"] = "  "
        self.write_manifest()
        self.assert_rejected("needs a valid split, match, source, and rights")

    def test_blank_rights_is_rejected(self):
        self.add_baseline()
        self.rows[1]["rights"] = ""
        self.write_manifest()
        self.assert_rejected("needs a valid split, match, source, and rights")

    def test_unknown_split_is_rejected(self):
        self.add_baseline()
        self.add("test", "q5", "event-e:Q5", b"test")
        self.write_manifest()
        self.assert_rejected("needs a valid split")

    def test_image_folder_must_match_declared_split(self):
        self.add_baseline()
        self.add("train", "q7", "event-g:Q7", b"misfiled")
        self.rows[-1]["split"] = "val"
        label = self.root / "labels" / "val" / "q7.txt"
        (self.root / "labels" / "train" / "q7.txt").replace(label)
        self.write_manifest()
        self.assert_rejected("row 4 has an invalid image path")

    def test_duplicate_row_is_rejected(self):
        self.add_baseline()
        self.rows.append(dict(self.rows[0]))
        self.write_manifest()
        self.assert_rejected("image appears twice in manifest")

    def test_row_with_extra_column_is_rejected(self):
        self.add_baseline()
        self.write_raw_manifest([
            ",".join(FIELDNAMES),
            "images/train/q1.jpg,train,event-a:Q1,https://example.org/video,owned,surplus",
            "images/val/q2.jpg,val,event-b:Q2,https://example.org/video,owned",
        ])
        self.assert_rejected("row 2 has extra or missing columns")

    def test_row_with_missing_column_is_rejected(self):
        self.add_baseline()
        self.write_raw_manifest([
            ",".join(FIELDNAMES),
            "images/train/q1.jpg,train,event-a:Q1,https://example.org/video,owned",
            "images/val/q2.jpg,val,event-b:Q2,https://example.org/video",
        ])
        self.assert_rejected("row 3 has extra or missing columns")

    def test_backslash_image_path_is_rejected(self):
        self.add_baseline()
        self.rows[0]["image"] = "images\\train\\q1.jpg"
        self.write_manifest()
        self.assert_rejected("must use forward-slash image paths")

    def test_drive_letter_image_path_is_rejected(self):
        self.add_baseline()
        self.rows[0]["image"] = "images/train/C:q1.jpg"
        self.write_manifest()
        self.assert_rejected("row 2 has an invalid image path")

    def test_absolute_drive_image_path_is_rejected(self):
        self.add_baseline()
        self.rows[0]["image"] = "C:/images/train/q1.jpg"
        self.write_manifest()
        self.assert_rejected("row 2 has an invalid image path")

    def test_non_image_suffix_is_rejected(self):
        self.add_baseline()
        self.add("train", "notes", "event-a:Q1", b"not an image", suffix=".txt")
        self.write_manifest()
        self.assert_rejected("row 4 has an invalid image path")

    def test_manifest_outside_dataset_root_is_rejected(self):
        self.add_baseline()
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        manifest = Path(outside.name) / "split_manifest.csv"
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "inside the dataset root"):
            validate_split_manifest(self.root, manifest)

    def test_empty_train_split_is_rejected(self):
        self.add("val", "q2", "event-b:Q2", b"val")
        self.add("holdout", "q3", "event-c:Q3", b"holdout")
        self.write_manifest()
        self.assert_rejected("must include labeled train and val")

    def test_empty_val_split_is_rejected(self):
        self.add("train", "q1", "event-a:Q1", b"train")
        self.add("holdout", "q3", "event-c:Q3", b"holdout")
        self.write_manifest()
        self.assert_rejected("must include labeled train and val")

    def test_cli_resolves_relative_manifest_against_dataset_root(self):
        self.add_baseline()
        self.write_manifest(self.root / "custom.csv")
        elsewhere = tempfile.TemporaryDirectory()
        self.addCleanup(elsewhere.cleanup)
        previous = Path.cwd()
        output = io.StringIO()
        try:
            os.chdir(elsewhere.name)
            with contextlib.redirect_stdout(output):
                main([str(self.root), "--manifest", "custom.csv"])
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(self.root), "--manifest", str(self.root / "custom.csv")])
        finally:
            os.chdir(previous)
        report = json.loads(output.getvalue())
        self.assertEqual(report["images"], {"train": 1, "val": 1, "holdout": 0})


if __name__ == "__main__":
    unittest.main()
