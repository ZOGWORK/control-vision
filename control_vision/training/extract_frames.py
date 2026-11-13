from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract(video: Path, output: Path, every_seconds: float = 2.0, max_frames: int | None = None) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release(); raise RuntimeError(f"Invalid FPS in {video}")
    interval = max(1, round(fps * every_seconds))
    output.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_index % interval == 0:
            target = output / f"{video.stem}-{frame_index:09d}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {target}")
            written += 1
            if max_frames and written >= max_frames: break
        frame_index += 1
    cap.release()
    return written


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
