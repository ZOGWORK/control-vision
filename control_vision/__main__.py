import subprocess
import sys
from pathlib import Path

from control_vision.app.main_window import run


def smoke_test() -> None:
    """Exercise bundled resources and start the Qt window without showing it."""
    import imageio_ffmpeg
    from PySide6.QtWidgets import QApplication
    from rapidocr_onnxruntime import RapidOCR

    from control_vision.app.main_window import MainWindow

    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    resources = ["LICENSE", "licenses/GPL-3.0.txt", "docs/BIOBUZZ_RULES.md",
                 "docs/THIRD_PARTY_NOTICES.md",
                 "control_vision/config/scoring_biobuzz.json"]
    if getattr(sys, "frozen", False):
        resources += ["SOURCE.txt", "FFMPEG_LICENSE.txt", "FFMPEG_BUILD.txt", "FFMPEG_SOURCE.txt"]
    for relative in resources:
        if not (root / relative).is_file():
            raise RuntimeError(f"Missing packaged resource: {relative}")
    app = QApplication([])
    window = MainWindow()
    try:
        if "BIOBUZZ" not in window.build_rules_reference():
            raise RuntimeError("Bundled BIOBUZZ rule reference is unavailable")
        RapidOCR()
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"],
                       capture_output=True, check=True, timeout=10)
        app.processEvents()
    finally:
        window.close()
        app.quit()

if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        run()
