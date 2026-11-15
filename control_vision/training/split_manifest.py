"""Audit match-level dataset splits before training or release validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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


def parse_source_match(value: str) -> tuple[str, str]:
    """Split ``EVENT:MATCH`` into a normalised (event, match) pair.

    Case, whitespace and leading zeros in match numbers are ignored, so
    ``USWASAM1 : q03`` and ``uswasam1:Q3`` name the same match.
    """
    event, separator, match = value.partition(":")
    event = " ".join(event.split()).casefold()
    match = "".join(match.split()).casefold()
    match = re.sub(r"\d+", lambda digits: str(int(digits.group())), match)
    if not separator or not event or not match:
        raise ValueError(f"source_match must look like EVENT:MATCH, got {value!r}")
    return event, match


def validate_split_manifest(root: Path, manifest: Path | None = None) -> dict:
    """Reject mislabeled or leaking splits; return a provenance summary.

    A relative ``manifest`` is resolved against ``root``.
    """
    root = root.resolve()
    if manifest is None:
        manifest = root / "split_manifest.csv"
    elif not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest.resolve()
    if not manifest.is_file() or not manifest.is_relative_to(root):
        raise ValueError("split manifest must be a file inside the dataset root")

    rows: dict[str, int] = {split: 0 for split in SPLITS}
    matches: dict[tuple[str, str], str] = {}
    events: dict[str, set[str]] = {}
    images: set[Path] = set()
    labels: set[Path] = set()
    digests: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"manifest needs columns: {', '.join(sorted(FIELDS))}")
        for line_number, row in enumerate(reader, 2):
            if None in row or any(row[name] is None for name in FIELDS):
                raise ValueError(f"manifest row {line_number} has extra or missing columns")
            split = row["split"].strip()
            if split not in SPLITS or not row["source"].strip() or not row["rights"].strip():
                raise ValueError(f"manifest row {line_number} needs a valid split, match, source, and rights")
            try:
                match = parse_source_match(row["source_match"])
            except ValueError as exc:
                raise ValueError(f"manifest row {line_number}: {exc}") from None
            previous = matches.setdefault(match, split)
            if previous != split:
                raise ValueError(f"match {':'.join(match)!r} appears in both {previous} and {split}")
            events.setdefault(match[0], set()).add(split)

            image_ref = row["image"].strip()
            if "\\" in image_ref:
                raise ValueError(f"manifest row {line_number} must use forward-slash image paths")
            relative = PurePosixPath(image_ref)
            if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3
                    or relative.parts[:2] != ("images", split) or relative.suffix.lower() not in IMAGE_SUFFIXES
                    or any(":" in part for part in relative.parts)):
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
            labels.add(label.resolve())
            image_hash = _sha256(image)
            prior_split = digests.setdefault(image_hash, split)
            if prior_split != split:
                raise ValueError(f"identical image content appears in {prior_split} and {split}: {image_ref}")
            rows[split] += 1

    for event, splits in sorted(events.items()):
        if "holdout" in splits and len(splits) > 1:
            others = " and ".join(sorted(splits - {"holdout"}))
            raise ValueError(f"holdout event {event!r} also appears in {others}")

    image_root = root / "images"
    if image_root.is_dir():
        unlisted = sorted(path for path in image_root.rglob("*")
                          if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                          and path.resolve() not in images)
        if unlisted:
            raise ValueError(f"{len(unlisted)} image(s) are absent from the manifest: {unlisted[0]}")
    label_root = root / "labels"
    if label_root.is_dir():
        orphans = sorted(path for path in label_root.rglob("*.txt")
                         if path.is_file() and path.resolve() not in labels)
        if orphans:
            raise ValueError(f"{len(orphans)} label file(s) have no manifest image: {orphans[0]}")
    if not rows["train"] or not rows["val"]:
        raise ValueError("manifest must include labeled train and val images")
    return {"images": rows, "matches": len(matches), "events": len(events),
            "holdout_events": sorted(event for event, splits in events.items() if "holdout" in splits),
            "manifest_sha256": _sha256(manifest)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check match-level split leakage and label provenance.")
    parser.add_argument("dataset", type=Path, help="dataset root containing split_manifest.csv")
    parser.add_argument("--manifest", type=Path,
                        help="CSV manifest inside dataset root; a relative path is resolved against the dataset root")
    args = parser.parse_args(argv)
    try:
        result = validate_split_manifest(args.dataset, args.manifest)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
