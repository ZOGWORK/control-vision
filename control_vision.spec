# PyInstaller spec for a Windows one-folder build. Models and videos remain external.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
import imageio_ffmpeg

root = Path(SPECPATH)
datas = collect_data_files("control_vision", includes=["config/*.yaml", "config/*.json"])
datas += collect_data_files("rapidocr_onnxruntime")
datas += [(str(root / "docs" / "BIOBUZZ_RULES.md"), "docs")]
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
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ControlVision", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ControlVision")
