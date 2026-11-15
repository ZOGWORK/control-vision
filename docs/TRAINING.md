# Local training workflow

1. Download only BIOBUZZ footage confirmed as a fixed full-field camera. Do not use the listed holdout event footage for training.
2. Extract candidate frames with `python -m control_vision.training.extract_frames path\to\match.mp4 --output datasets\biobuzz-v1\images\unlabeled`. Each video gets its own folder, named after the video.
3. Label boxes in CVAT and export YOLO format. Split by event, not by adjacent frames, into `images/train`, `images/val`, `labels/train`, and `labels/val` (and `images/holdout` for holdout frames). Keep each video's folder whole when you move it into a split. Copy `control_vision/config/data_biobuzz.example.yaml` to the dataset and adjust its path. An absolute `path` is safest; a relative path is rejected if it could name different datasets from the launch directory and the YAML directory. The training command passes the validated absolute path to Ultralytics. Use class order `robot`, `pollen`, `nectar`, or train only `robot` during the prototype. Class 0 must be `robot`.
4. Install the pinned training packages in a separate environment: `python -m pip install -r requirements-train.txt`. The command checks that `ultralytics`, `onnx`, `onnxruntime`, and `PyYAML` import before it reads the dataset, and names the missing package if one does not.
5. Use a local base checkpoint (no model download is attempted): `python -m control_vision.training.train_yolo --data datasets\biobuzz-v1\data.yaml --base-weights models\yolo11n.pt --version 1.0.0 --device cpu`. The version must be three numbers, such as `1.0.0`. Files are published to the repository's `models/` folder and Ultralytics runs go to the repository's `runs/training/`, whatever the launch directory; use `--models-dir` and `--runs-dir` to choose other folders.

## Checks before training

The command refuses to start unless:

- `data.yaml` is a mapping whose `names` are `robot` or `robot`, `pollen`, `nectar`, with IDs 0, 1, 2 and no gaps.
- The version is unused and not locked by another run, and no incomplete files exist for it.
- Every train/val image is readable and has a YOLO label file, and boxes and class IDs are valid. Empty label files are allowed for negative images.
- Both the train and val splits contain labeled robots.
- No identical image file appears in both train and val.
- No folder directly under `images/train`, `images/val`, or `images/holdout` appears in more than one of them. `extract_frames` writes one such folder per video, so this catches frames from one match placed in two splits.
- If `split_manifest.csv` exists at the dataset root, it passes `python -m control_vision.training.split_manifest` (see `docs/DATASET_SPLITS.md`). The manifest also lists every image under `images/`, so move `images/unlabeled` out of the dataset before using one. The manifest is optional for now.

These checks cannot detect visually similar frames from one match saved under different folder names or encodings. Split whole events by venue and date, and use the split manifest to record which match each image came from.

## Training settings

Ultralytics is told not to install packages or use the network (`YOLO_AUTOINSTALL=False`, `YOLO_OFFLINE=True`), and training plots are off because they download a font. The starting settings follow `docs/BUILD_BRIEF.md`: image size 640, 100 epochs, batch 16, `lr0` 0.01, patience 20. Colour augmentation is light because colour separates POLLEN from NECTAR and the two alliances: `hsv_h` 0.0, `hsv_s` 0.2, `hsv_v` 0.2. These are a starting point. Tune them on the val split, and the version record keeps the full training configuration.

## Publishing

After training, the command validates the best checkpoint on the val split and records overall and per-class mAP50 and mAP50-95. It refuses to publish if validation returns no metrics or no robot metrics. It exports ONNX at opset 12 or newer, then checks the file in ONNX Runtime on CPU: it runs one blank image, checks that the output has 4 + number-of-classes channels, and checks that the class names stored in the ONNX file match `data.yaml`.

The model is published as `models/yolo11n-biobuzz-v<version>.onnx` and `models/yolo11n-biobuzz-v<version>.json`. Both are written to temporary files first, then put in place without overwriting anything; the JSON record is written last and stores the ONNX file's SHA-256. A version counts as published only when both files exist and the hash matches. Existing versions are never overwritten. The record format is described in `models/README.md`.

## If a run is interrupted

- A run holds `models/.yolo11n-biobuzz-v<version>.lock`, which records its process ID and start time. If a run is killed, the lock stays and the next run for that version stops with the lock path. Check that no training process with that ID is still running, then delete the lock file.
- If the ONNX file exists without a matching JSON record, the command reports the version as incomplete and names the files. That version was never published. Delete the named files and train it again, or use a new version.
- A killed run can also leave hidden `.yolo11n-biobuzz-v<version>.*.partial` files in `models/`. They are never used and can be deleted when no run is active.

The official BIOBUZZ Manual TU09 (Nov 12, 2026) describes POLLEN as yellow, about 2.8 inches, and NECTAR as red or blue, about 3.6 inches. The suggested class list is `robot`, `pollen`, and `nectar`; alliance color is a NECTAR attribute, not an extra class. The app uses class 0 (`robot`) for tracking. If classes 1 and 2 are present, changes in their detected count inside a user-drawn image-space ROI can create flagged review candidates. That count signal does not establish the scoring achievement, ownership, or robot attribution. Validation metrics come from the val split; they do not establish holdout performance. Keep holdout events completely separate and report metrics with the specific matches used.
