# PyInstaller spec for a Windows one-folder build. Models and videos remain external.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
import imageio_ffmpeg

root = Path(SPECPATH)
datas = collect_data_files("control_vision", includes=["config/*.yaml", "config/*.json"])
datas += collect_data_files("rapidocr_onnxruntime")
datas += [(str(root / "docs" / "BIOBUZZ_RULES.md"), "docs")]
release_dir = root / "build" / "release"
required_release_files = ["SOURCE.txt", "FFMPEG_LICENSE.txt", "FFMPEG_BUILD.txt"]
missing = [name for name in required_release_files if not (release_dir / name).is_file()]
if missing:
    raise RuntimeError("Run scripts/build_windows.ps1 first; missing release metadata: " + ", ".join(missing))
datas += [(str(root / "LICENSE"), "."),
          (str(root / "licenses" / "GPL-3.0.txt"), "licenses"),
          (str(root / "README.md"), "."),
          (str(root / "docs" / "THIRD_PARTY_NOTICES.md"), "docs")]
datas += [(str(release_dir / name), ".") for name in required_release_files]
ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()

a = Analysis(
    [str(root / "control_vision" / "__main__.py")],
    pathex=[str(root)],
    binaries=[(ffmpeg_binary, "imageio_ffmpeg/binaries")],
    datas=datas,
    hiddenimports=["ultralytics.trackers.bot_sort", "ultralytics.trackers.byte_tracker", "onnxruntime", "rapidocr_onnxruntime"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
# PyInstaller's generic ctypes scan can pick up ICU 78 from the host's Poppler
# distribution. This Qt wheel uses the Windows ICU ABI available on supported
# Windows hosts; bundling that unrelated ICU causes QtCore error 127 at import.
a.binaries = [entry for entry in a.binaries
              if Path(entry[0]).name.casefold() not in {"icuuc.dll", "icudt78.dll"}]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ControlVision", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ControlVision")
