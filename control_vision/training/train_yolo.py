from __future__ import annotations

import argparse
import hashlib
import json
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
    model = YOLO(str(args.base_weights))
    run = model.train(data=str(args.data), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                      lr0=0.01, patience=20, device=args.device, project="runs/training", name=f"biobuzz-{args.version}")
    trainer = model.trainer
    metrics = getattr(trainer, "metrics", {})
    best_pt = Path(getattr(trainer, "best", Path(trainer.save_dir) / "weights" / "best.pt"))
    if not best_pt.is_file(): raise SystemExit(f"Training completed without a best checkpoint at {best_pt}")
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
    onnx_path = models_dir / f"yolo11n-biobuzz-v{args.version}.onnx"
    metadata_path = models_dir / f"yolo11n-biobuzz-v{args.version}.json"
    if onnx_path.exists() or metadata_path.exists():
        raise SystemExit(f"Refusing to overwrite existing model version {args.version}.")
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
    data_root = args.data.resolve().parent
    record = {"version": args.version, "date": date.today().isoformat(), "dataset_sha256": dataset_hash(data_root),
              "classes": names_list, "base_weights": str(args.base_weights), "training_config": {"epochs": args.epochs, "batch": args.batch,
              "imgsz": args.imgsz, "lr0": 0.01, "patience": 20, "device": args.device}, "validation_metrics": metrics,
              "onnx": onnx_path.name, "onnxruntime": "smoke_checked"}
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {onnx_path} and {metadata_path}")


if __name__ == "__main__":
    main()
