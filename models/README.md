# Model weights

No trained Control Vision weights are included. No weights are shipped in this repository or downloaded by the app; `*.onnx` and `*.pt` files are ignored by git. Train a YOLO11n model locally with `docs/TRAINING.md`.

Do not use a generic COCO model as if it were trained for FTC robots. The app expects the configured model's class `0` to represent `robot`. The remaining BIOBUZZ classes may be `pollen` and `nectar`, as described in Manual TU09; class color notes live in `control_vision/config/game_classes_biobuzz.json`. Keep old versions; never overwrite a released weight.

## Version records

Each trained version is a pair of files:

- `yolo11n-biobuzz-v<X.Y.Z>.onnx`: the exported model.
- `yolo11n-biobuzz-v<X.Y.Z>.json`: its record, written after the ONNX file.

A version is published only when the JSON record exists and its `onnx_sha256` matches the ONNX file. An ONNX file without a matching record is an incomplete artifact from an interrupted run; see `docs/TRAINING.md`.

The JSON record contains:

| Field | Meaning |
| --- | --- |
| `version` | Three-number version, such as `1.0.0`. |
| `date` | Date the version was published. |
| `dataset_sha256` | SHA-256 of the train, val, and holdout images and labels plus `data.yaml`, computed before training. `images/unlabeled` is not included. |
| `dataset_root`, `data_yaml` | Where the dataset was read from. |
| `split_manifest_sha256` | SHA-256 of `split_manifest.csv`, or `null` when the dataset has none. |
| `classes` | Class names in ID order; class 0 is `robot`. |
| `base_weights`, `base_weights_sha256` | Starting checkpoint and its SHA-256. |
| `best_checkpoint` | Local path of the best training checkpoint that was exported. |
| `training_config` | Every training setting passed to Ultralytics, including image size, epochs, batch, learning rate, patience, and HSV augmentation. |
| `validation_metrics` | Val-split metrics: `overall` and `per_class` mAP50 and mAP50-95 (`null` for a class with no val labels), plus Ultralytics' `summary`. |
| `export` | ONNX export settings, including the requested opset. |
| `onnxruntime_check` | Result of the ONNX Runtime CPU check: opset found in the file, output shape, and `smoke_run`. |
| `ultralytics_version` | Ultralytics version used for training. |
| `onnx`, `onnx_sha256` | ONNX file name and its SHA-256. |

Add a line for each published version below, with its record's date, dataset hash, and val metrics, and the matches the metrics were measured on.

## ReID weights

OSNet ReID weights are also not included. BoT-SORT can use an optional local ReID checkpoint selected in Settings; without one it logs that appearance matching is disabled and uses motion/camera compensation. A ReID file must be compatible with the installed Ultralytics encoder. No weights are downloaded.
