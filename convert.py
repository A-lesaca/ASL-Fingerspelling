"""Convert a folder-per-class image dataset into a landmark feature set."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


from detector import HandDetector  
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="folder-per-class root")
    ap.add_argument("--output", type=Path, required=True, help=".npz to write")
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument(
        "--skip",
        nargs="*",
        default=["J", "Z"],
        help="classes to exclude (default: the two motion letters)",
    )
    args = ap.parse_args()

    class_dirs = sorted(d for d in args.input.iterdir() if d.is_dir())
    class_dirs = [d for d in class_dirs if d.name not in set(args.skip)]
    if not class_dirs:
        raise SystemExit(f"no class folders found under {args.input}")

    X: list[np.ndarray] = []
    y: list[str] = []
    attempted = 0

    with HandDetector(static_image_mode=True, min_detection_confidence=0.4) as det:
        for cdir in class_dirs:
            files = sorted(p for p in cdir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if args.limit_per_class:
                files = files[: args.limit_per_class]

            found = 0
            for path in files:
                attempted += 1
                img = cv2.imread(str(path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Dataset images are ordinary photographs, not selfie-flipped.
                hit = det.detect(rgb, frame_is_mirrored=False)
                if hit is None:
                    continue
                X.append(hit.features)
                y.append(cdir.name)
                found += 1

            rate = found / len(files) if files else 0.0
            flag = "  <-- low" if rate < 0.5 else ""
            print(f"{cdir.name:>6}: {found:5d}/{len(files):5d} detected ({rate:5.1%}){flag}")

    if not X:
        raise SystemExit("no hands detected in any image -- check the input path")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=np.stack(X), y=np.array(y))
    print(f"\nwrote {len(X)} samples ({len(X)/attempted:.1%} of images) -> {args.output}")


if __name__ == "__main__":
    main()