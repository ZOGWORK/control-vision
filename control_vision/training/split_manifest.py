"""Audit match-level dataset splits before training or release validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


SPLITS = ("train", "val", "holdout")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FIELDS = {"image", "split", "source_match", "source", "rights"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_manifest(root: Path, manifest: Path | None = None) -> dict:
    """Reject mislabeled or leaking splits; return a provenance summary."""
    root = root.resolve()
    manifest = (manifest or root / "split_manifest.csv").resolve()
    if not manifest.is_file() or not manifest.is_relative_to(root):
        raise ValueError("split manifest must be a file inside the dataset root")

    rows: dict[str, int] = {split: 0 for split in SPLITS}
    matches: dict[str, str] = {}
    images: set[Path] = set()
    digests: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"manifest needs columns: {', '.join(sorted(FIELDS))}")
        for line_number, row in enumerate(reader, 2):
            if None in row or any(row[name] is None for name in FIELDS):
                raise ValueError(f"manifest row {line_number} has extra or missing columns")
            split = row["split"].strip()
            match = row["source_match"].strip().casefold()
            if split not in SPLITS or not match or not row["source"].strip() or not row["rights"].strip():
                raise ValueError(f"manifest row {line_number} needs a valid split, match, source, and rights")
            previous = matches.setdefault(match, split)
            if previous != split:
                raise ValueError(f"match {match!r} appears in both {previous} and {split}")

            image_ref = row["image"].strip()
            if "\\" in image_ref:
                raise ValueError(f"manifest row {line_number} must use forward-slash image paths")
            relative = PurePosixPath(image_ref)
            if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3
                    or relative.parts[:2] != ("images", split) or relative.suffix.lower() not in IMAGE_SUFFIXES
                    or ":" in relative.parts[0]):
                raise ValueError(f"manifest row {line_number} has an invalid image path")
            image = (root / Path(*relative.parts)).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError(f"manifest row {line_number} image is missing or outside the dataset: {image_ref}")
            if image in images:
                raise ValueError(f"image appears twice in manifest: {image_ref}")
            images.add(image)
            label = root / "labels" / split / Path(*relative.parts[2:]).with_suffix(".txt")
            if not label.resolve().is_relative_to(root) or not label.is_file():
                raise ValueError(f"missing label for {image_ref}: {label}")
            image_hash = _sha256(image)
            prior_split = digests.setdefault(image_hash, split)
            if prior_split != split:
                raise ValueError(f"identical image content appears in {prior_split} and {split}: {image_ref}")
            rows[split] += 1

    for split in SPLITS:
        image_dir = root / "images" / split
        if not image_dir.is_dir():
            continue
        unlisted = [path for path in image_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.resolve() not in images]
        if unlisted:
            raise ValueError(f"{len(unlisted)} {split} image(s) are absent from the manifest: {unlisted[0]}")
    if not rows["train"] or not rows["val"]:
        raise ValueError("manifest must include labeled train and val images")
    return {"images": rows, "matches": len(matches), "manifest_sha256": _sha256(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check match-level split leakage and label provenance.")
    parser.add_argument("dataset", type=Path, help="dataset root containing split_manifest.csv")
    parser.add_argument("--manifest", type=Path, help="CSV manifest inside dataset root")
    args = parser.parse_args()
    try:
        result = validate_split_manifest(args.dataset, args.manifest)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
