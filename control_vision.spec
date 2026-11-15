# PyInstaller spec for a Windows one-folder build. Run it through
# scripts/build_windows.ps1, which writes the release metadata it requires.
# Trained detector weights and match videos remain external to the bundle.
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata
import imageio_ffmpeg

root = Path(SPECPATH)
sys.path.insert(0, str(root / "scripts"))
from bundle_licenses import write_license_bundle  # noqa: E402

datas = collect_data_files("control_vision", includes=["config/*.yaml", "config/*.json"])
# RapidOCR's config.yaml and its three ONNX text models.
datas += collect_data_files("rapidocr_onnxruntime")
# Ultralytics reads installed package versions before loading a model.
for distribution in ("onnxruntime", "torch", "torchvision", "ultralytics", "numpy", "opencv-python"):
    datas += copy_metadata(distribution)
datas += [(str(root / "docs" / "BIOBUZZ_RULES.md"), "docs")]
release_dir = root / "build" / "release"
required_release_files = ["SOURCE.txt", "FFMPEG_LICENSE.txt", "FFMPEG_BUILD.txt", "FFMPEG_SOURCE.txt"]
missing = [name for name in required_release_files if not (release_dir / name).is_file()]
if missing:
    raise RuntimeError("Run scripts/build_windows.ps1 first; missing release metadata: " + ", ".join(missing))
datas += [(str(root / "LICENSE"), "."),
          (str(root / "README.md"), "."),
          (str(root / "docs" / "THIRD_PARTY_NOTICES.md"), "docs")]
datas += [(str(path), "licenses") for path in sorted((root / "licenses").glob("*.txt"))]
datas += [(str(release_dir / name), ".") for name in required_release_files]
ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()
# torchvision loads its C++ ops (_C_stable.pyd, image_stable.pyd) with
# torch.ops.load_library from its own folder rather than importing them, so
# PyInstaller's import analysis misses them. Without them Ultralytics' NMS
# fails with "operator torchvision::nms does not exist".
torchvision_dir = Path(importlib.util.find_spec("torchvision").origin).parent
torchvision_native = [(str(path), "torchvision") for path in sorted(torchvision_dir.glob("*.pyd"))
                      + sorted(torchvision_dir.glob("*.dll"))]
if not any(Path(source).name.startswith("_C") for source, _ in torchvision_native):
    raise RuntimeError(f"torchvision C++ ops were not found in {torchvision_dir}")

a = Analysis(
    [str(root / "control_vision" / "__main__.py")],
    pathex=[str(root)],
    binaries=[(ffmpeg_binary, "imageio_ffmpeg/binaries")] + torchvision_native,
    datas=datas,
    hiddenimports=["control_vision.smoke_test", "ultralytics.trackers.bot_sort", "ultralytics.trackers.byte_tracker",
                   "onnxruntime", "rapidocr_onnxruntime"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Export and training-only packages: not in requirements.txt, but present
    # in training environments. Excluding them keeps local and CI builds alike.
    excludes=["onnx", "onnxslim", "tensorboard", "tensorflow"],
    noarchive=False,
    optimize=1,
)
# PyInstaller's generic ctypes scan can pick up ICU 78 from the host's Poppler
# distribution. This Qt wheel uses the Windows ICU ABI available on supported
# Windows hosts; bundling that unrelated ICU causes QtCore error 127 at import.
a.binaries = [entry for entry in a.binaries
              if Path(entry[0]).name.casefold() not in {"icuuc.dll", "icudt78.dll"}]
# License files for every Python distribution actually collected above.
collected = [entry[0] for entry in a.pure] + [entry[0] for entry in a.binaries] + [entry[0] for entry in a.datas]
license_dir = release_dir / "licenses" / "python-packages"
for source, folder in write_license_bundle(collected, license_dir, always=["pyinstaller"]):
    destination = "licenses/python-packages" if folder == "." else f"licenses/python-packages/{folder}"
    a.datas.append((f"{destination}/{Path(source).name}", source, "DATA"))
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ControlVision", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ControlVision")
