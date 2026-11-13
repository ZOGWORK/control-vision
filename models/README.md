# Model weights

No trained Control Vision weights are included yet. Add a locally trained and validated YOLO11n ONNX model here using the versioned naming and metadata format in `docs/BUILD_BRIEF.md`.

Do not use a generic COCO model as if it were trained for FTC robots. The app expects the configured model's class `0` to represent `robot`. Record the model version, date, dataset hash, training configuration, and validation metrics when weights are added. Keep old versions; never overwrite a released weight.

OSNet ReID weights are also not included. Phase 1's current BoT-SORT configuration uses motion and camera compensation without appearance embeddings until a vetted local ReID model is selected and packaged.
