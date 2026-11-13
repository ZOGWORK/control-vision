from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import date
from pathlib import Path

import cv2
import numpy as np


def dataset_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".txt", ".jpg", ".jpeg", ".png"}):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _dataset_root(data: Path, config: dict) -> Path:
    configured_value = config.get("path", ".")
    if not isinstance(configured_value, str):
        raise ValueError("path must name a dataset directory in data.yaml")
    configured = Path(configured_value)
    if configured.is_absolute():
        return configured
    # The example data.yaml uses a path relative to the launch directory.
    # Also accept a path relative to the YAML for portable datasets.
    from_cwd = configured.resolve()
    from_yaml = (data.resolve().parent / configured).resolve()
    train_ref = config.get("train", "")
    if isinstance(train_ref, str) and (from_cwd / train_ref).is_dir():
        return from_cwd
    return from_yaml


def validate_dataset(data: Path, config: dict, class_count: int) -> Path:
    """Reject broken YOLO labels and exact train/val leakage before training."""
    root = _dataset_root(data, config)
    seen_hashes: dict[str, str] = {}
    train_robot_count = 0
    for split in ("train", "val"):
        image_ref = config.get(split)
        if not isinstance(image_ref, str):
            raise ValueError(f"{split} must name an images directory in data.yaml")
        images_dir = Path(image_ref)
        if not images_dir.is_absolute():
            images_dir = root / images_dir
        if not images_dir.is_dir() or images_dir.parent.name != "images":
            raise ValueError(f"{split} images directory is missing or outside images/: {images_dir}")
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        images = sorted(path for path in images_dir.iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError(f"{split} has no images in {images_dir}")
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"Missing YOLO label file for {split} image {image.name}: {label}")
            if cv2.imread(str(image)) is None:
                raise ValueError(f"Unreadable {split} image: {image}")
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            previous = seen_hashes.get(image_hash)
            if previous is not None and previous != split:
                raise ValueError(f"Identical image appears in train and val: {image.name}")
            seen_hashes[image_hash] = split
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
                if split == "train" and class_id == 0:
                    train_robot_count += 1
    if train_robot_count == 0:
        raise ValueError("Training split has no labeled robots (class 0)")
    return root


def main():
    parser = argparse.ArgumentParser(description="Train locally from a hand-labeled BIOBUZZ dataset and export versioned ONNX weights.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml with train/val splits")
    parser.add_argument("--base-weights", type=Path, required=True, help="Existing local YOLO11n .pt weights; no download is attempted")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.data.is_file() or not args.base_weights.is_file():
        parser.error("Both --data and --base-weights must refer to existing local files.")
    try:
        import yaml
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the pinned requirements before training.") from exc
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = config.get("names", {})
    names_list = [names[index] if isinstance(names, dict) else names[index] for index in range(len(names))]
    if names_list not in (["robot"], ["robot", "pollen", "nectar"]):
        parser.error("Supported classes are ['robot'] or ['robot','pollen','nectar'] in that order; alliance color is an attribute, not a class.")
    models_dir = Path("models")
    onnx_path = models_dir / f"yolo11n-biobuzz-v{args.version}.onnx"
    metadata_path = models_dir / f"yolo11n-biobuzz-v{args.version}.json"
    if onnx_path.exists() or metadata_path.exists():
        parser.error(f"Refusing to overwrite existing model version {args.version}.")
    try:
        data_root = validate_dataset(args.data, config, len(names_list))
    except ValueError as exc:
        parser.error(str(exc))
    model = YOLO(str(args.base_weights))
    run = model.train(data=str(args.data), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                      lr0=0.01, patience=20, device=args.device, project="runs/training", name=f"biobuzz-{args.version}")
    trainer = model.trainer
    metrics = run.results_dict if hasattr(run, "results_dict") else run if isinstance(run, dict) else {}
    best_pt = Path(getattr(trainer, "best", Path(trainer.save_dir) / "weights" / "best.pt"))
    if not best_pt.is_file(): raise SystemExit(f"Training completed without a best checkpoint at {best_pt}")
    models_dir.mkdir(exist_ok=True)
    trained = YOLO(str(best_pt))
    exported = Path(trained.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=False))
    shutil.copy2(exported, onnx_path)
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        blank = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        session.run(None, {input_info.name: blank})
    except Exception as exc:
        onnx_path.unlink(missing_ok=True)
        raise SystemExit(f"ONNX Runtime smoke check failed: {exc}") from exc
    record = {"version": args.version, "date": date.today().isoformat(), "dataset_sha256": dataset_hash(data_root),
              "classes": names_list, "base_weights": str(args.base_weights), "training_config": {"epochs": args.epochs, "batch": args.batch,
              "imgsz": args.imgsz, "lr0": 0.01, "patience": 20, "device": args.device}, "validation_metrics": metrics,
              "onnx": onnx_path.name, "onnxruntime": "smoke_checked"}
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {onnx_path} and {metadata_path}")


if __name__ == "__main__":
    main()
