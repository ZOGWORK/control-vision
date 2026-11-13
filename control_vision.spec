# PyInstaller spec for a Windows one-folder build. Models and videos remain external.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH)
datas = collect_data_files("control_vision", includes=["config/*.yaml", "config/*.json"])
datas += [(str(root / "docs" / "BIOBUZZ_RULES.md"), "docs")]

a = Analysis(
    [str(root / "control_vision" / "__main__.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=["ultralytics.trackers.bot_sort", "ultralytics.trackers.byte_tracker", "onnxruntime"],
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
