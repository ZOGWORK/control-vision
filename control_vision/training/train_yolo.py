from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, Callable, Iterator

import cv2
import numpy as np

from control_vision.image_files import read_image

# IMAGE_SUFFIXES is Ultralytics' image list, shared with the manifest audit so
# it covers every image that is trained on. Decoding is checked below on the
# actual training machine before an image enters a dataset hash or split.
from control_vision.training.split_manifest import IMAGE_SUFFIXES, validate_split_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+")
MIN_OPSET = 12
# Class 0 is always robot; every other class is a scoring element. The
# four-class schema (v0.4.0 on) splits elements by colour so a detector can
# tell yellow POLLEN from each alliance's NECTAR.
SUPPORTED_CLASSES = (["robot"], ["robot", "pollen", "nectar"],
                     ["robot", "yellow_pollen", "red_nectar", "blue_nectar"])
# Import name, pip distribution name. Checked before any training starts.
TRAINING_PACKAGES = (("yaml", "PyYAML"), ("ultralytics", "ultralytics"),
                     ("onnx", "onnx"), ("onnxruntime", "onnxruntime"))
# Starting point from docs/BUILD_BRIEF.md. Colour separates POLLEN from
# NECTAR and red from blue, so hue is not shifted and saturation/value move
# only lightly. Tune these on the val split and record each change.
TRAINING_DEFAULTS: dict[str, Any] = {"lr0": 0.01, "patience": 20, "hsv_h": 0.0, "hsv_s": 0.2, "hsv_v": 0.2,
                                     # Plots fetch a font from the internet on first use.
                                     "plots": False}
# --fine-tune: continue from trained BIOBUZZ weights. optimizer "auto" ignores lr0, so name the optimizer to make the
# low learning rate take effect; a short warmup; mosaic off for the last epochs; stop sooner without improvement.
FINE_TUNE_DEFAULTS: dict[str, Any] = {"optimizer": "AdamW", "lr0": 0.0005, "lrf": 0.1, "warmup_epochs": 1,
                                      "close_mosaic": 5, "patience": 8}
REQUIRED_RECORD_FIELDS = ("version", "date", "dataset_sha256", "classes", "training_config", "validation_metrics",
                          "split_provenance")
MANIFEST_NAME = "split_manifest.csv"
AUDITED, EXPLORATORY = "match_audited", "exploratory_unverified"


def model_stem(version: str) -> str:
    return f"yolo11n-biobuzz-v{version}"


def check_version(version: str) -> None:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Model version must be three numbers separated by dots, like 1.0.0 (got {version!r})")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_names(config: object) -> list[str]:
    """Return data.yaml class names in ID order, or raise a plain ValueError."""
    if not isinstance(config, dict):
        raise ValueError("data.yaml must be a mapping with path, train, val, and names entries")
    names = config.get("names")
    if isinstance(names, dict):
        if not names or set(names) != set(range(len(names))) or any(isinstance(key, bool) for key in names):
            raise ValueError("names in data.yaml must use class IDs 0, 1, 2, ... without gaps; class 0 must be robot")
        values = [names[index] for index in range(len(names))]
    elif isinstance(names, list):
        values = names
    else:
        raise ValueError("data.yaml needs a names list or mapping; class 0 must be robot")
    if values not in SUPPORTED_CLASSES:
        raise ValueError("Supported classes are ['robot'], ['robot','pollen','nectar'] or "
                         "['robot','yellow_pollen','red_nectar','blue_nectar'] in that order; "
                         "a robot's alliance color is an attribute, not a class.")
    return list(values)


def _dataset_root(data: Path, config: dict) -> Path:
    configured_value = config.get("path", ".")
    if not isinstance(configured_value, str):
        raise ValueError("path must name a dataset directory in data.yaml")
    configured = Path(configured_value)
    if configured.is_absolute():
        return configured.resolve()
    train_ref = config.get("train", "")
    if isinstance(train_ref, str) and Path(train_ref).is_absolute():
        return Path(train_ref).resolve().parent.parent
    # The example data.yaml uses a path relative to the launch directory.
    # Also accept a path relative to the YAML for portable datasets.
    from_cwd = configured.resolve()
    from_yaml = (data.resolve().parent / configured).resolve()
    cwd_has_train = isinstance(train_ref, str) and (from_cwd / train_ref).is_dir()
    yaml_has_train = isinstance(train_ref, str) and (from_yaml / train_ref).is_dir()
    if cwd_has_train and yaml_has_train and from_cwd != from_yaml:
        raise ValueError("Ambiguous relative dataset path; use an absolute path in data.yaml")
    if cwd_has_train:
        return from_cwd
    return from_yaml


def _split_images_dir(root: Path, config: dict, split: str) -> Path:
    image_ref = config.get(split)
    if not isinstance(image_ref, str):
        raise ValueError(f"{split} must name an images directory in data.yaml")
    images_dir = Path(image_ref)
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    images_dir = images_dir.resolve()
    if not images_dir.is_dir() or images_dir.parent.name != "images" or images_dir.parent.parent != root:
        raise ValueError(f"{split} images directory is missing or outside images/: {images_dir}")
    return images_dir


def _split_dirs(root: Path, config: dict) -> dict[str, Path]:
    """Image folders for train, val, and (when present) images/holdout."""
    root = root.resolve()
    dirs ={split: _split_images_dir(root, config, split) for split in ("train", "val")}
    holdout = (root / "images" / "holdout").resolve()
    if holdout.is_dir() and holdout not in dirs.values():
        dirs["holdout"] = holdout
    return dirs


def _check_source_folders(split_dirs: dict[str, Path]) -> None:
    """Reject a per-video frame folder (as written by extract_frames) in two splits."""
    owners: dict[str, str] = {}
    for split, images_dir in split_dirs.items():
        for child in sorted(images_dir.iterdir()):
            if child.is_dir():
                previous = owners.setdefault(child.name.casefold(), split)
                if previous != split:
                    raise ValueError(f"Source folder {child.name!r} appears in both images/{previous} and "
                                     f"images/{split}; keep every frame from one video in one split")


def dataset_hash(root: Path, config: dict, data_file: Path) -> str:
    """SHA-256 of the split images, their labels, and data.yaml (not images/unlabeled)."""
    root = root.resolve()
    entries: list[tuple[str, Path]] = []
    for images_dir in _split_dirs(root, config).values():
        labels_dir = root / "labels" / images_dir.name
        for folder, keep in ((images_dir, lambda p: p.suffix.lower() in IMAGE_SUFFIXES),
                             (labels_dir, lambda p: p.suffix.lower() == ".txt")):
            if folder.is_dir():
                entries += [(path.relative_to(root).as_posix(), path) for path in folder.rglob("*")
                            if path.is_file() and keep(path)]
    entries.sort()
    entries.append(("<data.yaml>", data_file))
    digest = hashlib.sha256()
    for name, path in entries:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _check_duplicate(seen_hashes: dict[str, str], image: Path, split: str) -> None:
    image_hash = _sha256(image)
    previous = seen_hashes.setdefault(image_hash, split)
    if previous != split:
        raise ValueError(f"Identical image appears in {previous} and {split}: {image.name}")


def validate_dataset(data: Path, config: dict, class_count: int) -> Path:
    """Reject broken YOLO labels and train/val leakage before training."""
    root = _dataset_root(data, config)
    split_dirs = _split_dirs(root, config)
    _check_source_folders(split_dirs)
    seen_hashes: dict[str, str] = {}
    robot_counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        images_dir = split_dirs[split]
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        images = sorted(path for path in images_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"{split} has no images in {images_dir}")
        for image in images:
            label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing YOLO label file for {split} image {image.name}: {label}")
            if read_image(image) is None:
                raise ValueError(f"Unreadable {split} image: {image}")
            _check_duplicate(seen_hashes, image, split)
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"{label}:{line_number}: expected class x_center y_center width height")
                try:
                    class_id = int(values[0])
                    x, y, width, height = map(float, values[1:])
                except ValueError as exc:
                    raise ValueError(f"{label}:{line_number}: invalid class or box number") from exc
                if not (0 <= class_id < class_count and all(math.isfinite(v) for v in (x, y, width, height))
                        and 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1
                        and x - width / 2 >= -1e-6 and x + width / 2 <= 1 + 1e-6
                        and y - height / 2 >= -1e-6 and y + height / 2 <= 1 + 1e-6):
                    raise ValueError(f"{label}:{line_number}: class or normalized box is out of range")
                if class_id == 0:
                    robot_counts[split] += 1
    holdout = split_dirs.get("holdout")
    if holdout is not None:
        # Holdout frames are only compared by content; their labels are checked at evaluation time.
        for image in sorted(path for path in holdout.rglob("*")
                            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
            _check_duplicate(seen_hashes, image, "holdout")
    if robot_counts["train"] == 0:
        raise ValueError("Training split has no labeled robots (class 0)")
    if robot_counts["val"] == 0:
        raise ValueError("Validation split has no labeled robots (class 0), so robot mAP cannot be measured")
    return root


def training_split_provenance(root: Path, config: dict, exploratory: bool = False) -> dict:
    """Audit split_manifest.csv at the dataset root and return the record's split_provenance.

    Release training requires the manifest. With ``exploratory`` and no
    manifest the run is marked unverified; a manifest that exists is always
    validated, so the flag cannot bypass an invalid one.
    """
    root = root.resolve()
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        if exploratory:
            return {"mode": EXPLORATORY, "manifest_sha256": None}
        raise ValueError(f"{MANIFEST_NAME} is missing from the dataset root {root}; release training needs it "
                         f"(see docs/DATASET_SPLITS.md), or pass --exploratory-without-manifest for an "
                         f"unverified local run")
    for split in ("train", "val"):
        images_dir = _split_images_dir(root, config, split)
        if images_dir != (root / "images" / split).resolve():
            raise ValueError(f"{split} in data.yaml must be images/{split} so {MANIFEST_NAME} covers the "
                             f"images that are trained on (got {images_dir})")
    # The audit rejects any image under images/ that the manifest does not list,
    # using the same IMAGE_SUFFIXES that select training images.
    try:
        audit = validate_split_manifest(root, manifest)
    except ValueError as exc:
        raise ValueError(f"{MANIFEST_NAME}: {exc}") from None
    return {"mode": AUDITED, "manifest_sha256": audit["manifest_sha256"], "matches": audit["matches"],
            "events": audit["events"], "holdout_events": audit["holdout_events"], "images": audit["images"]}


def train_list(root: Path, config: dict, list_file: Path) -> tuple[list[Path], dict]:
    """The images named by a --train-list file, and what the record says about it.

    The list chooses how often each training image is used (a line per use;
    repeats oversample). data.yaml still names images/train, so the manifest
    audit, label checks and dataset hash cover the same images as before; the
    list may only name existing images inside that folder.
    """
    root = root.resolve()
    train_dir = _split_images_dir(root, config, "train")
    try:
        # Read once: the recorded hash and counts describe exactly these bytes, which are what is trained on.
        raw = list_file.read_bytes()
        text = raw.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read the train list {list_file}: {exc}") from None
    images: list[Path] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        entry = line.strip()
        if not entry:
            continue
        image = Path(entry)
        image = (image if image.is_absolute() else root / image).resolve()
        if not image.is_relative_to(train_dir):
            raise ValueError(f"{list_file.name}:{line_number}: {entry} is not inside the training images "
                             f"directory {train_dir}")
        if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
            raise ValueError(f"{list_file.name}:{line_number}: {entry} is not an existing training image")
        images.append(image)
    if not images:
        raise ValueError(f"The train list {list_file.name} names no images")
    unique = set(images)
    labels_dir = train_dir.parent.parent / "labels" / train_dir.name
    # The same rule as validate_dataset applies to what is actually trained on: some robots (class 0).
    with_robots = sum(1 for image in unique
                      if any(line.split()[:1] == ["0"] for line in
                             (labels_dir / image.relative_to(train_dir).with_suffix(".txt"))
                             .read_text(encoding="utf-8").splitlines()))
    if not with_robots:
        raise ValueError(f"The train list {list_file.name} names no image with a labeled robot (class 0)")
    in_train = sum(1 for path in train_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return images, {"file": str(list_file.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "lines": len(images),
                    "unique_images": len(unique), "robot_images": with_robots, "train_images": in_train,
                    "train_images_not_listed": in_train - len(unique)}


def model_version_state(models_dir: Path, version: str) -> tuple[str, str]:
    """Return ("absent" | "published" | "incomplete", plain explanation).

    A version is published only when its JSON record exists, names its ONNX
    file, and records that file's SHA-256. Publishing writes the JSON last.
    """
    stem = model_stem(version)
    onnx_path = models_dir / f"{stem}.onnx"
    metadata_path = models_dir / f"{stem}.json"
    if not onnx_path.exists() and not metadata_path.exists():
        return "absent", ""
    problem = None
    if not metadata_path.is_file():
        problem = "its JSON record is missing"
    else:
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        if not isinstance(record, dict) or record.get("onnx") != onnx_path.name \
                or not isinstance(record.get("onnx_sha256"), str):
            problem = "its JSON record is unreadable or does not record the ONNX SHA-256"
        elif not onnx_path.is_file():
            problem = "its ONNX file is missing"
        elif _sha256(onnx_path) != record["onnx_sha256"]:
            problem = "its ONNX file does not match the SHA-256 in its JSON record"
    if problem is None:
        return "published", ""
    existing = ", ".join(str(path) for path in (onnx_path, metadata_path) if path.exists())
    return "incomplete", (f"Model version {version} is an incomplete artifact and was never published ({problem}): "
                          f"{existing}. If no training run is active, delete these files and train the version "
                          f"again, or choose a new version.")


def _lock_message(lock_path: Path, version: str) -> str:
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        detail = f"PID {owner['pid']}, started {owner['started']}"
    except (OSError, ValueError, KeyError, TypeError):
        detail = "owner details unreadable"
    return (f"Model version {version} is locked by {lock_path} ({detail}). Another run may still be training it; "
            f"if that process is no longer running, delete the lock file and retry.")


@contextmanager
def reserve_model_version(models_dir: Path, version: str) -> Iterator[tuple[Path, Path]]:
    """Reserve a version across processes and refuse existing or incomplete artifacts."""
    check_version(version)
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = model_stem(version)
    lock_path = models_dir / f".{stem}.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError(_lock_message(lock_path, version)) from None
    try:
        with lock:
            json.dump({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "version": version}, lock)
            lock.flush()
            os.fsync(lock.fileno())
        state, detail = model_version_state(models_dir, version)
        if state == "published":
            raise ValueError(f"Refusing to overwrite existing model version {version}")
        if state == "incomplete":
            raise ValueError(detail)
        yield models_dir / f"{stem}.onnx", models_dir / f"{stem}.json"
    finally:
        lock_path.unlink(missing_ok=True)


def _check_record(record: dict) -> None:
    missing = [field for field in REQUIRED_RECORD_FIELDS if not record.get(field)]
    if missing:
        raise ValueError(f"Refusing to publish a model record without {', '.join(missing)}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record["dataset_sha256"])):
        raise ValueError("Refusing to publish a model record without a SHA-256 dataset hash")
    if record["classes"][0] != "robot":
        raise ValueError("Refusing to publish a model whose class 0 is not robot")
    provenance = record["split_provenance"]
    if not isinstance(provenance, dict):
        provenance = {}
    mode = provenance.get("mode")
    if mode == AUDITED:
        counts = provenance.get("images")
        valid = bool(re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("manifest_sha256")))
                 and all(type(provenance.get(key)) is int for key in ("matches", "events"))
                 and isinstance(provenance.get("holdout_events"), list)
                 and isinstance(counts, dict) and all(type(counts.get(split)) is int for split in ("train", "val")))
    else:
        valid = mode == EXPLORATORY and provenance.get("manifest_sha256", "") is None
    if not valid:
        raise ValueError(f"Refusing to publish a model record without a valid split_provenance "
                         f"(mode {AUDITED!r} with its manifest audit, or {EXPLORATORY!r})")


def _write_partial(final: Path, write: Callable[[BinaryIO], None], partials: list[Path]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".partial", dir=final.parent)
    partials.append(Path(name))
    with os.fdopen(descriptor, "wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())


def _link_new(source: Path, target: Path) -> None:
    try:
        os.link(source, target)  # Fails instead of replacing an existing file.
    except FileExistsError:
        raise ValueError(f"Refusing to overwrite existing model file {target}") from None


def publish_model_artifacts(exported: Path, onnx_path: Path, metadata_path: Path, record: dict) -> dict:
    """Publish the ONNX file, then the JSON record that marks it complete.

    Both files are written and fsynced under temporary names first, then
    hard-linked into place, so neither final name ever holds partial bytes.
    Returns the record as written, including the ONNX SHA-256.
    """
    _check_record(record)
    partials: list[Path] = []
    created: list[Path] = []
    digest = hashlib.sha256()

    def copy_exported(target: BinaryIO) -> None:
        with exported.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)

    try:
        _write_partial(onnx_path, copy_exported, partials)
        final_record = {**record, "onnx": onnx_path.name, "onnx_sha256": digest.hexdigest()}
        encoded = (json.dumps(final_record, indent=2) + "\n").encode("utf-8")

        def write_record(target: BinaryIO) -> None:
            target.write(encoded)

        _write_partial(metadata_path, write_record, partials)
        _link_new(partials[0], onnx_path)
        created.append(onnx_path)
        _link_new(partials[1], metadata_path)
        created.append(metadata_path)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in partials:
            path.unlink(missing_ok=True)
    return final_record


def check_training_packages() -> dict[str, ModuleType]:
    """Disable Ultralytics network installs, then import every training package."""
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    modules = {}
    for module_name, package in TRAINING_PACKAGES:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            raise SystemExit(f"The {package} package is missing; install the training packages with "
                             f"python -m pip install -r requirements-train.txt") from None
    return modules


def _finite(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number):
        raise ValueError(f"Validation returned no usable {what}; the model was not published")
    return number


def validation_metrics(results: Any, names: list[str]) -> dict:
    """Per-class and overall mAP50 / mAP50-95 from an Ultralytics val() result."""
    summary = getattr(results, "results_dict", None)
    box = getattr(results, "box", None)
    class_index = getattr(results, "ap_class_index", None)
    if class_index is None:
        class_index = getattr(box, "ap_class_index", None)
    if not isinstance(summary, dict) or not summary or box is None or class_index is None:
        raise ValueError("Validation returned no metrics; the model was not published")
    ap50, ap = getattr(box, "ap50", None), getattr(box, "ap", None)
    per_class: dict[str, dict[str, float] | None] = {name: None for name in names}
    try:
        for position, class_id in enumerate(list(class_index)):
            if 0 <= int(class_id) < len(names):
                per_class[names[int(class_id)]] = {"mAP50": _finite(ap50[position], "per-class mAP50"),  # type: ignore[index]
                                                   "mAP50-95": _finite(ap[position], "per-class mAP50-95")}  # type: ignore[index]
    except (TypeError, IndexError):
        raise ValueError("Validation returned no per-class metrics; the model was not published") from None
    if per_class["robot"] is None:
        raise ValueError("Validation returned no robot (class 0) metrics; the model was not published")
    return {"split": "val",
            "overall": {"mAP50": _finite(getattr(box, "map50", None), "mAP50"),
                        "mAP50-95": _finite(getattr(box, "map", None), "mAP50-95")},
            "per_class": per_class,
            "summary": {str(key): _finite(value, str(key)) for key, value in summary.items()}}


def check_onnx_model(path: Path, names: list[str], imgsz: int, onnx: Any, ort: Any) -> dict:
    """Check opset, run one ONNX Runtime CPU inference, and check outputs and class names."""
    try:
        proto = onnx.load(str(path))
        opset = max((entry.version for entry in proto.opset_import if entry.domain in ("", "ai.onnx")), default=0)
    except Exception as exc:
        raise ValueError(f"Could not read the exported ONNX file {path}: {exc}") from exc
    if opset < MIN_OPSET:
        raise ValueError(f"Exported ONNX uses opset {opset}; opset {MIN_OPSET} or newer is required")
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        outputs = session.run(None, {input_info.name: np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)})
        metadata = dict(session.get_modelmeta().custom_metadata_map)
    except Exception as exc:
        raise ValueError(f"ONNX Runtime smoke check failed: {exc}") from exc
    shape = tuple(np.asarray(outputs[0]).shape) if outputs else ()
    channels = 4 + len(names)
    if len(shape) != 3 or shape[0] != 1 or shape[1] != channels or shape[2] < 1:
        raise ValueError(f"ONNX output shape {shape} does not match 1 x {channels} x N "
                         f"(4 box values + {len(names)} classes)")
    raw_names = metadata.get("names")
    try:
        parsed = ast.literal_eval(raw_names) if isinstance(raw_names, str) else None
    except (ValueError, SyntaxError):
        parsed = None
    if not isinstance(parsed, dict) or parsed != dict(enumerate(names)):
        raise ValueError(f"ONNX metadata class names {raw_names!r} do not match {names}; class 0 must be robot")
    return {"opset": opset, "output_shape": list(shape), "provider": "CPUExecutionProvider", "smoke_run": "passed"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train locally from a hand-labeled BIOBUZZ dataset and export versioned ONNX weights.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml with train/val splits")
    parser.add_argument("--base-weights", type=Path, required=True, help="Existing local YOLO11n .pt weights; no download is attempted")
    parser.add_argument("--version", default="1.0.0", help="Model version, three numbers such as 1.0.0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models",
                        help="Where versioned ONNX files and JSON records are published (default: the repo's models/)")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs" / "training",
                        help="Where Ultralytics writes training and validation runs (default: the repo's runs/training)")
    parser.add_argument("--exploratory-without-manifest", action="store_true",
                        help="Allow a local run without split_manifest.csv; its record says exploratory_unverified "
                             "and it is not a release model. A manifest that exists is still validated.")
    parser.add_argument("--train-list", type=Path,
                        help="Text file of training images to use, one per line (repeat a line to oversample). Each "
                             "must be an existing image inside data.yaml's images/train, which is still audited.")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Continue from trained BIOBUZZ weights with a low fixed learning rate "
                             f"({', '.join(f'{k}={v}' for k, v in FINE_TUNE_DEFAULTS.items())})")
    args = parser.parse_args(argv)
    try:
        check_version(args.version)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.data.is_file() or not args.base_weights.is_file():
        parser.error("Both --data and --base-weights must refer to existing local files.")
    modules = check_training_packages()
    yaml, onnx, ort = modules["yaml"], modules["onnx"], modules["onnxruntime"]
    YOLO = modules["ultralytics"].YOLO
    models_dir, runs_dir = args.models_dir.resolve(), args.runs_dir.resolve()
    with ExitStack() as stack:
        # Dataset, class, and version problems are usage errors. Errors raised
        # by Ultralytics or torch during training keep their traceback.
        try:
            try:
                config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError(f"data.yaml is not valid YAML: {exc}") from None
            names = class_names(config)
            # Provenance first: a missing or invalid manifest stops the run before images are decoded.
            split_provenance = training_split_provenance(_dataset_root(args.data, config), config,
                                                         args.exploratory_without_manifest)
            data_root = validate_dataset(args.data, config, len(names))
            dataset_sha256 = dataset_hash(data_root, config, args.data)
            listed, list_record = (train_list(data_root, config, args.train_list) if args.train_list
                                   else (None, None))
            onnx_path, metadata_path = stack.enter_context(reserve_model_version(models_dir, args.version))
        except ValueError as exc:
            parser.error(str(exc))
        exploratory = split_provenance["mode"] == EXPLORATORY
        if exploratory:
            print(f"UNVERIFIED EXPLORATORY RUN: {data_root} has no {MANIFEST_NAME}, so match-level split separation "
                  f"is not checked. The record will say {EXPLORATORY}; this is not a release model.", flush=True)
        train_args = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz, "device": args.device,
                      **TRAINING_DEFAULTS, **(FINE_TUNE_DEFAULTS if args.fine_tune else {})}
        # Ultralytics resolves a relative `path` against its process CWD.
        # Pass the exact dataset root that was validated.
        temp = stack.enter_context(tempfile.TemporaryDirectory(prefix="control-vision-data-"))
        normalized_data = Path(temp) / "data.yaml"
        normalized = {**config, "path": str(data_root)}
        if listed is not None:
            # Only now, after every check ran on images/train, does training read the list.
            listed_file = Path(temp) / "train_list.txt"
            listed_file.write_text("".join(f"{image}\n" for image in listed), encoding="utf-8")
            normalized["train"] = str(listed_file)
        normalized_data.write_text(yaml.safe_dump(normalized), encoding="utf-8")
        model = YOLO(str(args.base_weights))
        model.train(data=str(normalized_data), project=str(runs_dir), name=f"biobuzz-{args.version}", **train_args)
        trainer = model.trainer
        best_pt = Path(getattr(trainer, "best", Path(trainer.save_dir) / "weights" / "best.pt"))
        if not best_pt.is_file():
            raise SystemExit(f"Training completed without a best checkpoint at {best_pt}")
        trained = YOLO(str(best_pt))
        results = trained.val(data=str(normalized_data), split="val", imgsz=args.imgsz, batch=args.batch,
                              device=args.device, plots=False, project=str(runs_dir),
                              name=f"biobuzz-{args.version}-val")
        try:
            metrics = validation_metrics(results, names)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        exported = Path(trained.export(format="onnx", imgsz=args.imgsz, opset=MIN_OPSET, simplify=False))
        try:
            onnx_check = check_onnx_model(exported, names, args.imgsz, onnx, ort)
            record = {"version": args.version, "date": date.today().isoformat(), "dataset_sha256": dataset_sha256,
                      "dataset_root": str(data_root), "data_yaml": str(args.data.resolve()),
                      "split_provenance": split_provenance, "classes": names, "base_weights": str(args.base_weights),
                      "base_weights_sha256": _sha256(args.base_weights), "best_checkpoint": str(best_pt),
                      "training_config": {"model": "yolo11n", **train_args},
                      "fine_tune": args.fine_tune, "train_list": list_record,
                      "validation_metrics": metrics,
                      "export": {"format": "onnx", "imgsz": args.imgsz, "requested_opset": MIN_OPSET,
                                 "simplify": False},
                      "onnxruntime_check": onnx_check,
                      "ultralytics_version": getattr(modules["ultralytics"], "__version__", None)}
            publish_model_artifacts(exported, onnx_path, metadata_path, record)
        except ValueError as exc:
            raise SystemExit(f"Model version {args.version} was not published: {exc}") from None
        print(f"Saved {onnx_path} and {metadata_path}")
        if exploratory:
            print(f"UNVERIFIED: version {args.version} is {EXPLORATORY} (no {MANIFEST_NAME}); "
                  f"do not use or report it as a release model.")

if __name__ == "__main__":
    main()
