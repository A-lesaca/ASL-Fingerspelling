"""Transcribe ASL fingerspelling from a pre-recorded video file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

from detector import HandDetector, draw_landmarks
from smoothing import NO_HAND, LetterDebouncer


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int]:
    """Open a video file and report its frame rate and length."""
    if not path.exists():
        raise SystemExit(f"video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not decode {path} -- is it a valid video file?")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # missing or NaN in some containers
        print("warning: no frame rate in file metadata, assuming 30fps")
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, float(fps), total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True, help="input video file")
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--output", type=Path, default=None,
                    help="write the transcript here as well as to stdout")
    ap.add_argument("--min-confidence", type=float, default=0.60)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--min-votes", type=int, default=9)
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame (2 halves the runtime)")
    ap.add_argument("--mirrored", action="store_true",
                    help="set if the clip was recorded in selfie/mirror view")
    ap.add_argument("--show", action="store_true",
                    help="display annotated frames while processing")
    ap.add_argument("--quiet", action="store_true", help="suppress progress")
    args = ap.parse_args()

    model_path = args.models / "model.joblib"
    if not model_path.exists():
        raise SystemExit(
            f"no model at {model_path} -- run train.py first"
        )
    model = joblib.load(model_path)
    # Labels come from the estimator, so they cannot disagree with its output.
    classes = list(model.classes_)

    cap, fps, total = open_video(args.video)
    frame_ms = 1000.0 / fps
    if not args.quiet:
        length = f"{total} frames" if total else "unknown length"
        print(f"{args.video.name}: {fps:.2f} fps, {length}", file=sys.stderr)

    deb = LetterDebouncer(
        window=args.window,
        min_votes=args.min_votes,
        min_confidence=args.min_confidence,
    )

    frame_no = 0
    detected = 0
    committed: list[tuple[float, str]] = []

    # VIDEO running mode, with timestamps taken from the file's own frame rate
    # rather than a synthetic clock -- see HandDetector.detect.
    with HandDetector() as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if args.every > 1 and frame_no % args.every:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hit = det.detect(
                rgb,
                frame_is_mirrored=args.mirrored,
                timestamp_ms=int(frame_no * frame_ms),
            )

            if hit is None:
                deb.update(NO_HAND, 0.0)
                label, conf = None, 0.0
            else:
                detected += 1
                probs = model.predict_proba(hit.features[None, :])[0]
                k = int(probs.argmax())
                label, conf = classes[k], float(probs[k])
                if deb.update(label, conf) is not None:
                    committed.append((frame_no / fps, deb.text[-1]))

            if args.show:
                if hit is not None:
                    draw_landmarks(frame, hit.raw)
                caption = f"{label or '-'} {conf:.0%}"
                cv2.putText(frame, caption, (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 190, 255), 2)
                cv2.putText(frame, deb.text[-40:], (12, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imshow("transcribing (q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if not args.quiet and total and frame_no % 100 == 0:
                print(f"  {frame_no}/{total} frames", end="\r", file=sys.stderr)

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    processed = frame_no // max(args.every, 1)
    if not args.quiet:
        rate = detected / processed if processed else 0.0
        print(f"\nprocessed {processed} frames, hand found in {rate:.1%}",
              file=sys.stderr)
        if rate < 0.5:
            print("  low detection rate -- check lighting, framing, and "
                  "whether --mirrored is set correctly", file=sys.stderr)
        if committed:
            print("\ntimeline:", file=sys.stderr)
            for seconds, char in committed:
                shown = {" ": "<space>"}.get(char, char)
                print(f"  {seconds:6.2f}s  {shown}", file=sys.stderr)
        print("\ntranscript:", file=sys.stderr)

    print(deb.text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(deb.text + "\n")
        if not args.quiet:
            print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()