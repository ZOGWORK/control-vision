# Third-party notices for the Windows bundle

Control Vision is licensed under GNU AGPL-3.0; see `LICENSE` in the bundle.
The corresponding Control Vision source commit is identified in `SOURCE.txt`.

- [Ultralytics](https://github.com/ultralytics/ultralytics) is distributed under AGPL-3.0 in the pinned open-source dependency. No trained robot weights are bundled.
- [Qt for Python / PySide6](https://doc.qt.io/qtforpython-6/licenses.html) is distributed under its community LGPLv3/GPLv3 terms, with additional third-party components described by Qt. The Windows build uses the PySide6 wheels listed in `requirements.txt` at the source commit.
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) is BSD 2-Clause licensed. Its Windows wheel supplies the separate FFmpeg executable in this bundle.
- [FFmpeg](https://ffmpeg.org/legal.html) is free software. The actual bundled binary's license statement is in `FFMPEG_LICENSE.txt`, its version and configure options are in `FFMPEG_BUILD.txt`, and its SHA-256 and exact upstream source links are in `FFMPEG_SOURCE.txt`. The pinned Windows wheel's binary reports GPL version 3 or later; the license text is in `licenses/GPL-3.0.txt`.

The package versions are fixed in `requirements.txt` at the Control Vision source commit. Other Python packages and native libraries in the bundle retain their respective licenses. Consult their package metadata and upstream notices for full terms.
