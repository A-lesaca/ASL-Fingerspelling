"""Live webcam demo: sign at the camera, watch text appear."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aslfs.detector import HandDetector  
from aslfs.smoothing import NO_HAND, LetterDebouncer 

GREEN, WHITE, AMBER = (80, 220, 100), (255, 255, 255), (60, 190, 255)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.80)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--min-votes", type=int, default=9)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.models / "model.keras")
    classes = json.loads((args.models / "classes.json").read_text())

    deb = LetterDebouncer(
        window=args.window,
        min_votes=args.min_votes,
        min_confidence=args.min_confidence,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    fps_times: deque[float] = deque(maxlen=30)
    drawing = mp.solutions.drawing_utils
    connections = mp.solutions.hands.HAND_CONNECTIONS

    with HandDetector() as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hit = det.detect(rgb, frame_is_mirrored=True)

            if hit is None:
                deb.update(NO_HAND, 0.0)
                label, conf = "-", 0.0
            else:
                probs = model.predict(hit.features[None, :], verbose=0)[0]
                k = int(probs.argmax())
                label, conf = classes[k], float(probs[k])
                deb.update(label, conf)
                drawing.draw_landmarks(frame, hit.raw, connections)

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
            cv2.putText(frame, deb.text[-38:] or "...", (12, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)


            cand, frac = deb.progress()
            if cand:
                cv2.rectangle(frame, (12, h - 22),
                              (12 + int(min(frac, 1.0) * 260), h - 12), GREEN, -1)
                cv2.putText(frame, cand, (285, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

            fps_times.append(time.perf_counter() - t0)
            fps = 1.0 / max(np.mean(fps_times), 1e-6)
            cv2.putText(frame, f"{label} {conf:.0%}   {fps:4.1f} fps", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2)

            cv2.imshow("ASL fingerspelling (c=clear, q=quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                deb.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()