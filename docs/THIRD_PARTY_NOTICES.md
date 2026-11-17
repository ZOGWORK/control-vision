# Third-party notices for the Windows bundle

Control Vision is licensed under the GNU Affero General Public License v3.0;
see `LICENSE` in the bundle. `SOURCE.txt` names the exact Control Vision
source commit the bundle was built from, with a link to that source.

The bundle contains no trained robot detector or ReID weights. Users supply
their own model files and match videos, which stay outside the application
folder. The only model files shipped are RapidOCR's text-recognition models.

## Where the license texts are

- `LICENSE`: Control Vision, AGPL-3.0.
- `licenses/GPL-3.0.txt`, `licenses/LGPL-3.0.txt`, `licenses/Apache-2.0.txt`:
  full texts referenced below.
- `licenses/python-packages/INDEX.txt`: every Python distribution PyInstaller
  collected into this build, with its version, declared license and the
  license, notice and third-party files copied from the installed package into
  `licenses/python-packages/<name>-<version>/`. This folder also holds the
  Python runtime's own license. It is generated at build time, so it lists
  exactly what the build contains.
- `FFMPEG_LICENSE.txt`, `FFMPEG_BUILD.txt`, `FFMPEG_SOURCE.txt`: the bundled
  FFmpeg executable's own license statement, version and configure options, and
  its SHA-256 and upstream source.

## Main components

Versions are pinned in `requirements.txt` at the source commit.

- **Qt for Python (PySide6, shiboken6) and Qt 6**: used under the GNU LGPL v3
  (`licenses/LGPL-3.0.txt`, which supplements `licenses/GPL-3.0.txt`). Qt's
  libraries are shipped as separate, unmodified DLLs in `_internal/PySide6`, so
  they can be replaced with other compatible builds. The PySide6 wheels only
  carry Qt's commercial-license reference file; the community licensing is
  described at <https://doc.qt.io/qtforpython-6/licenses.html> and the
  third-party components inside Qt at <https://doc.qt.io/qt-6/licenses-used-in-qt.html>.
  Qt source: <https://download.qt.io/official_releases/qt/> and
  <https://code.qt.io/cgit/pyside/pyside-setup.git/>.
- **Ultralytics**: AGPL-3.0, the same license as Control Vision
  (<https://github.com/ultralytics/ultralytics>).
- **PySide6-Fluent-Widgets** (`qfluentwidgets`, 1.11.3): GPL-3.0 (`licenses/GPL-3.0.txt`),
  combinable with Control Vision's AGPL-3.0; the author also sells a commercial license
  (<https://github.com/zhiyiYo/PyQt-Fluent-Widgets>). Its dependencies: **PySideSix-Frameless-Window**
  (LGPL-3.0), **pywin32** (PSF), **darkdetect** (BSD-3-Clause).
- **PyTorch and torchvision**: BSD-style licenses. The copied `torch` files
  include PyTorch's `LICENSE` and the licenses of its bundled third-party
  components; the package declares the combined license expression shown in
  `INDEX.txt`.
- **ONNX Runtime**: MIT, with `ThirdPartyNotices.txt` copied from the package.
- **RapidOCR (rapidocr-onnxruntime)**: Apache-2.0 (`licenses/Apache-2.0.txt`).
  The package does not ship a license file. Its bundled detection,
  classification and recognition models (`ch_PP-OCRv4_det_infer.onnx`,
  `ch_ppocr_mobile_v2.0_cls_infer.onnx`, `ch_PP-OCRv4_rec_infer.onnx`) are
  ONNX conversions of PaddleOCR models, which PaddlePaddle publishes under
  Apache-2.0 (<https://github.com/PaddlePaddle/PaddleOCR>,
  <https://github.com/RapidAI/RapidOCR>).
- **OpenCV (opencv-python)**: Apache-2.0 for OpenCV and MIT for the wheel
  packaging. The wheel's `LICENSE-3RD-PARTY.txt` is copied; it covers the
  libraries built into OpenCV, including the LGPL FFmpeg build inside
  `opencv_videoio_ffmpeg*.dll` that OpenCV uses to decode video.
- **NumPy**: BSD-3-Clause. Its copied `LICENSE.txt` also covers the bundled
  OpenBLAS and compiler runtime libraries.
- **imageio-ffmpeg**: BSD-2-Clause. Its Windows wheel supplies the separate
  `ffmpeg.exe` used for highlight clips.
- **FFmpeg (the separate executable)**: the pinned binary reports GPL version 3
  or later (see `FFMPEG_LICENSE.txt` and `licenses/GPL-3.0.txt`); the configure
  options are in `FFMPEG_BUILD.txt` and the source in `FFMPEG_SOURCE.txt`. It is
  run as a separate program, not linked into Control Vision.
- **SQLite**: public domain (<https://sqlite.org/copyright.html>), built into
  the Python runtime's `sqlite3` module. Project files are SQLite databases.
- **Python**: PSF License Version 2, plus the components the Windows runtime
  includes, in `licenses/python-packages/python-*/LICENSE.txt`.
- **PyInstaller bootloader**: GPL-2.0-or-later with the PyInstaller bootloader
  exception, which allows distributing the bundled executable under any license.

Other Python packages and native libraries in the bundle keep their own
licenses, recorded in `licenses/python-packages/INDEX.txt`. Where a package
ships no license file, the index says so and gives the license it declares.
