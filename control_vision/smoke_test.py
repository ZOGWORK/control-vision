"""Non-interactive release check for the source tree and the frozen Windows build.

Run with ``ControlVision.exe --smoke-test --report smoke.json`` (or
``python -m control_vision --smoke-test``). It checks the bundled licenses,
configuration and OCR models, creates the Qt main window without showing it,
runs RapidOCR on a rendered image, runs the bundled FFmpeg to generate a short
H.264 clip, imports that clip into a temporary project database, and imports
Ultralytics. ``--model`` and ``--video`` additionally run Ultralytics tracking
with a local ONNX detector on a few frames of real footage.

The windowed executable has no console, so the JSON report file and the exit
code are the result: 0 when every step passed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

REQUIRED_DISTRIBUTIONS = ("PySide6", "opencv-python", "numpy", "ultralytics", "torch", "onnxruntime",
                          "rapidocr-onnxruntime", "imageio-ffmpeg")
OCR_MODELS = ("ch_PP-OCRv4_det_infer.onnx", "ch_PP-OCRv4_rec_infer.onnx", "ch_ppocr_mobile_v2.0_cls_infer.onnx")
CONFIG_FILES = ("scoring_biobuzz.json", "zones_biobuzz.json", "game_classes_biobuzz.json", "botsort.yaml")
# Trained-weight formats. Only RapidOCR's own models may ship in the bundle.
WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".torchscript", ".tflite", ".safetensors", ".ckpt",
                   ".mlmodel", ".mlpackage"}
ALLOWED_WEIGHT_DIR = ("rapidocr_onnxruntime", "models")
_RELEASE_FILES = ("SOURCE.txt", "FFMPEG_LICENSE.txt", "FFMPEG_BUILD.txt", "FFMPEG_SOURCE.txt",
                  "licenses/python-packages/INDEX.txt")
_OCR_TEXT = "2468"


class SmokeFailure(RuntimeError):
    pass


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise SmokeFailure(f"Missing packaged resource: {relative}")
    return path


def check_resources(root: Path, frozen: bool) -> dict[str, Any]:
    import yaml

    for relative in ("LICENSE", "licenses/GPL-3.0.txt", "licenses/LGPL-3.0.txt", "licenses/Apache-2.0.txt",
                     "docs/BIOBUZZ_RULES.md", "docs/THIRD_PARTY_NOTICES.md"):
        _require_file(root, relative)
    if "GNU AFFERO GENERAL PUBLIC LICENSE" not in _require_file(root, "LICENSE").read_text(encoding="utf-8"):
        raise SmokeFailure("LICENSE is not the AGPL-3.0 text")
    config_dir = root / "control_vision" / "config"
    for name in CONFIG_FILES:
        text = _require_file(config_dir, name).read_text(encoding="utf-8")
        parsed = yaml.safe_load(text) if name.endswith(".yaml") else json.loads(text)
        if not parsed:
            raise SmokeFailure(f"Configuration file {name} is empty")
    import rapidocr_onnxruntime
    ocr_dir = Path(rapidocr_onnxruntime.__file__).resolve().parent
    for name in OCR_MODELS:
        if (ocr_dir / "models" / name).stat().st_size < 100_000:
            raise SmokeFailure(f"RapidOCR model {name} is truncated")
    _require_file(ocr_dir, "config.yaml")
    result: dict[str, Any] = {"config": list(CONFIG_FILES), "ocr_models": list(OCR_MODELS)}
    if frozen:
        for relative in _RELEASE_FILES:
            _require_file(root, relative)
        source = (root / "SOURCE.txt").read_text(encoding="utf-8")
        match = re.search(r"^Commit: ([0-9a-f]{40})$", source, re.MULTILINE)
        if not match:
            raise SmokeFailure("SOURCE.txt does not record a 40-character source commit")
        result["source_commit"] = match.group(1)
        index = (root / "licenses" / "python-packages" / "INDEX.txt").read_text(encoding="utf-8")
        listed = {_canonical(name) for name in re.findall(r"^(\S+) \S+: ", index, re.MULTILINE)}
        missing = [name for name in REQUIRED_DISTRIBUTIONS if _canonical(name) not in listed]
        if missing:
            raise SmokeFailure("License index lacks bundled distributions: " + ", ".join(missing))
        _require_file(root, "PySide6/plugins/platforms/qwindows.dll")
        result["bundled_weights"] = find_bundled_weights(root)
        if result["bundled_weights"]:
            raise SmokeFailure("Model weights are bundled: " + ", ".join(result["bundled_weights"]))
    return result


def find_bundled_weights(root: Path) -> list[str]:
    """Relative paths of model-weight files other than RapidOCR's own models."""
    found = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if relative.parts[:2] == ALLOWED_WEIGHT_DIR:
            continue
        found.append(relative.as_posix())
    return sorted(found)


def check_qt_window() -> dict[str, Any]:
    from PySide6.QtWidgets import QApplication

    from control_vision.app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        if "BIOBUZZ" not in window.build_rules_reference():
            raise SmokeFailure("Bundled BIOBUZZ rule reference is unavailable")
        if not app.styleSheet() or app.property("control_vision_theme") not in {"light", "dark"}:
            raise SmokeFailure("The application theme was not applied")
        app.processEvents()
        return {"platform": app.platformName(), "title": window.windowTitle(),
                "theme": app.property("control_vision_theme")}
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def check_ocr() -> dict[str, Any]:
    import cv2
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    image = np.full((160, 480, 3), 255, dtype=np.uint8)
    cv2.putText(image, _OCR_TEXT, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 0), 8, cv2.LINE_AA)
    result, _ = engine(image)
    text = "".join(str(row[1]) for row in result or [])
    if _OCR_TEXT not in text.replace(" ", ""):
        raise SmokeFailure(f"RapidOCR read {text!r} from a rendered {_OCR_TEXT!r}")
    return {"text": text}


def check_ffmpeg(work: Path, root: Path, frozen: bool) -> tuple[dict[str, Any], Path]:
    import subprocess

    import imageio_ffmpeg

    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if frozen and root.resolve() not in executable.parents:
        raise SmokeFailure(f"FFmpeg resolved outside the bundle: {executable}")
    version = subprocess.run([str(executable), "-version"], capture_output=True, text=True, check=True, timeout=30)
    clip = work / "generated.mp4"
    subprocess.run([str(executable), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc2=size=1280x720:rate=30", "-t", "2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-y", str(clip)],
                   capture_output=True, text=True, check=True, timeout=120)
    if not clip.is_file() or clip.stat().st_size == 0:
        raise SmokeFailure("Bundled FFmpeg did not produce the generated test clip")
    return {"version": version.stdout.splitlines()[0], "clip_bytes": clip.stat().st_size}, clip


def check_project_import(work: Path, clip: Path) -> dict[str, Any]:
    import sqlite3

    from control_vision.pipeline.ingest import probe_video
    from control_vision.storage.database import ProjectDatabase

    metadata = probe_video(clip)
    if (metadata["width"], metadata["height"]) != (1280, 720) or abs(metadata["duration_s"] - 2.0) > 0.25:
        raise SmokeFailure(f"Generated clip probed as {metadata}")
    db_path = work / "project" / "control_vision.sqlite"
    db = ProjectDatabase(db_path)
    try:
        columns = {row["name"] for row in db.connection.execute("PRAGMA table_info(videos)")}
        if not {"analysis_status", "match_type", "tracking_mode"} <= columns:
            raise SmokeFailure("Project database migration did not add the current video columns")
        project_id = db.add_project("Smoke test")
        video_id = db.add_video(project_id, clip, metadata, "SMOKE", "Q1")
    finally:
        db.close()
    reopened = ProjectDatabase(db_path)  # Migration must be idempotent on an existing project.
    try:
        row = reopened.connection.execute("SELECT path, analysis_status FROM videos WHERE id=?", (video_id,)).fetchone()
    finally:
        reopened.close()
    if row is None or Path(row["path"]) != clip:
        raise SmokeFailure("Imported video was not saved in the project database")
    return {"sqlite": sqlite3.sqlite_version, "fps": metadata["fps"], "duration_s": metadata["duration_s"],
            "analysis_status": row["analysis_status"]}


def check_ultralytics() -> dict[str, Any]:
    import onnxruntime
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO  # noqa: F401  # The import itself is the check.
    from ultralytics.utils import AUTOINSTALL

    if is_frozen() and AUTOINSTALL:
        raise SmokeFailure("Ultralytics auto-install is enabled in the frozen build")
    # Ultralytics' detection post-processing runs torchvision's C++ NMS op,
    # which a frozen build only has if torchvision's native library shipped.
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]])
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5).tolist()
    if kept != [0, 2]:
        raise SmokeFailure(f"torchvision NMS kept {kept}, expected [0, 2]")
    return {"autoinstall": AUTOINSTALL, "torchvision": torchvision.__version__,"ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__, "providers": onnxruntime.get_available_providers()}


def check_tracking(root: Path, model: Path, video: Path, start_s: float, frames: int) -> dict[str, Any]:
    import cv2
    from ultralytics import YOLO

    tracker = root / "control_vision" / "config" / "botsort.yaml"
    detector = YOLO(str(model), task="detect")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise SmokeFailure(f"Could not open {video.name}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
        step = max(1, round(fps / 4))
        per_frame = []
        for index in range(frames):
            ok, frame = cap.read()
            if not ok:
                raise SmokeFailure(f"Could not read frame {index} of {video.name} from {start_s:.1f} s")
            started = time.perf_counter()
            result = detector.track(frame, persist=True, tracker=str(tracker), imgsz=640, conf=0.25, iou=0.5,
                                    verbose=False)[0]
            ids = [] if result.boxes.id is None else [int(value) for value in result.boxes.id.tolist()]
            per_frame.append({"boxes": len(result.boxes), "track_ids": ids,
                              "ms": round(1000 * (time.perf_counter() - started), 1)})
            for _ in range(step - 1):
                cap.grab()
    finally:
        cap.release()
    if not per_frame:
        raise SmokeFailure("Tracking produced no frames")
    return {"model": model.name, "video": video.name, "start_s": start_s, "frames": per_frame,
            "names": list(detector.names.values())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root, frozen = resource_root(), is_frozen()
    report: dict[str, Any] = {"ok": False, "frozen": frozen, "root": str(root), "steps": {}}
    started = time.perf_counter()

    def step(name: str, action: Callable[[], Any]) -> Any:
        begun = time.perf_counter()
        try:
            value = action()
        except Exception as exc:
            report["steps"][name] = {"ok": False, "seconds": round(time.perf_counter() - begun, 2),
                                     "error": f"{type(exc).__name__}: {exc}"}
            raise
        detail = value[0] if isinstance(value, tuple) else value
        report["steps"][name] = {"ok": True, "seconds": round(time.perf_counter() - begun, 2), "detail": detail}
        return value

    try:
        with tempfile.TemporaryDirectory(prefix="control-vision-smoke-") as temp:
            work = Path(temp)
            step("resources", lambda: check_resources(root, frozen))
            step("qt_window", check_qt_window)
            step("ocr", check_ocr)
            _, clip = step("ffmpeg", lambda: check_ffmpeg(work, root, frozen))
            step("project_import", lambda: check_project_import(work, clip))
            step("ultralytics", check_ultralytics)
            if args.model or args.video:
                if not (args.model and args.video):
                    raise SmokeFailure("--model and --video must be given together")
                step("tracking", lambda: check_tracking(root, Path(args.model), Path(args.video),
                                                        args.start, args.frames))
        report["ok"] = True
    except Exception as exc:  # Report every failure; the windowed build has no console.
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    report["seconds"] = round(time.perf_counter() - started, 2)
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ControlVision --smoke-test", description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--report", type=Path, help="write the JSON result to this file")
    parser.add_argument("--model", help="local ONNX detector for an optional tracking check (never bundled)")
    parser.add_argument("--video", help="match video for the optional tracking check")
    parser.add_argument("--start", type=float, default=60.0, help="seconds into --video to start tracking")
    parser.add_argument("--frames", type=int, default=5, help="frames to track, four per second of video")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = run(args)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    if sys.stdout is not None:
        print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.exit(main(sys.argv[1:]))
