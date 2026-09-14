"""Record your own landmark samples from the webcam."""

from __future__ import annotations

import argparse
import string
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


from detector import HandDetector, draw_landmarks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--burst", type=int, default=30, help="frames recorded per keypress")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    if args.output.exists():
        prev = np.load(args.output, allow_pickle=True)
        X, y = list(prev["X"]), list(prev["y"])
        print(f"resuming from {args.output} ({len(X)} samples)")
    else:
        X, y = [], []

    skip = {"J", "Z"}
    valid = {c for c in string.ascii_uppercase} - skip

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    recording: str | None = None
    remaining = 0
    burst_sizes: list[tuple[str, int]] = []

    with HandDetector() as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # selfie view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hit = det.detect(rgb, frame_is_mirrored=True)

            if hit is not None:
                draw_landmarks(frame, hit.raw)

            if recording and remaining > 0:
                if hit is not None:
                    X.append(hit.features)
                    y.append(recording)
                    remaining -= 1
                cv2.putText(
                    frame, f"REC {recording}  {remaining} left", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )
                if remaining == 0:
                    burst_sizes.append((recording, args.burst))
                    recording = None
            else:
                counts = Counter(y)
                thin = sorted(c for c in valid | {"space", "del"} if counts[c] < 40)
                msg = f"total {len(X)}"
                if thin:
                    msg += f" | need more: {' '.join(thin[:8])}"
                cv2.putText(
                    frame, msg, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
                )

            cv2.imshow("collect (letter=record, -=undo, esc=quit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # esc
                break
            if key == ord("-"):  # undo
                if burst_sizes:
                    label, n = burst_sizes.pop()
                    n = min(n, len(X))
                    del X[-n:]
                    del y[-n:]
                    print(f"undid burst of {n} for {label}")
            elif key == 32:  # space bar
                recording, remaining = "space", args.burst
            elif key in (8, 127):  # backspace
                recording, remaining = "del", args.burst
            elif 32 < key < 127 and chr(key).upper() in valid:
                recording, remaining = chr(key).upper(), args.burst

    cap.release()
    cv2.destroyAllWindows()

    if X:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, X=np.stack(X), y=np.array(y))
        print(f"saved {len(X)} samples -> {args.output}")


if __name__ == "__main__":
    main()