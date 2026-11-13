from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import cv2


def extract(video: Path, output: Path, every_seconds: float = 2.0, max_frames: int | None = None) -> int:
    if not math.isfinite(every_seconds) or every_seconds <= 0:
        raise ValueError("--every-seconds must be positive and finite")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release(); raise RuntimeError(f"Invalid FPS in {video}")
    try:
        interval = max(1, round(fps * every_seconds))
        expected_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        output.mkdir(parents=True, exist_ok=True)
        video_key = hashlib.sha256(str(video.resolve()).encode("utf-8")).hexdigest()[:10]
        frame_index = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                if math.isfinite(expected_frames) and expected_frames > 0 and frame_index < round(expected_frames):
                    raise RuntimeError(f"Video ended before its declared frame count: {video}")
                break
            if frame_index % interval == 0:
                target = output / f"{video.stem}-{video_key}-{frame_index:09d}.jpg"
                if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write {target}")
                written += 1
                if max_frames is not None and written >= max_frames:
                    break
            frame_index += 1
        return written
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Extract frames for manual CVAT labeling; this tool does not assign labels.")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/biobuzz-v1/images/unlabeled"))
    parser.add_argument("--every-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    total = 0
    for video in args.videos:
        count = extract(video, args.output, args.every_seconds, args.max_frames)
        print(f"{video.name}: extracted {count} unlabeled frames")
        total += count
    print(f"Total: {total}. Move reviewed labeled data into train/val and update data.yaml before training.")


if __name__ == "__main__":
    main()
