"""ASL fingerspelling: record, train and transcribe, all in the browser.

This is the whole project. Run it and open the page:

    python app.py

The camera runs on a background thread; frames are classified, smoothed into
letters and streamed to the browser as MJPEG. The same page records training
samples and fits the model, so there are no separate collection or training
scripts to run first.

Reading the camera off the main thread is deliberate on macOS: an OpenCV
window must live on the main thread and tends to hang when it does not.
Serving frames to a browser avoids the native GUI entirely.
"""

from __future__ import annotations

import argparse
import string
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from detector import HandDetector, draw_landmarks
from smoothing import DELETE, NO_HAND, SPACE, LetterDebouncer
from training import (RECOMMENDED_PER_CLASS, counts, load_samples,
                      save_samples, train_model)

# J and Z are drawn with motion, which a single-frame classifier cannot
# represent, so they are left out of the alphabet everywhere.
MOTION_LETTERS = {"J", "Z"}
ALPHABET = [c for c in string.ascii_uppercase if c not in MOTION_LETTERS]
TEACHABLE = ALPHABET + [SPACE, DELETE]

app = Flask(__name__)


class Engine:
    """Owns the camera thread and every piece of state it touches."""

    def __init__(self, models: Path, data: Path, camera: int) -> None:
        self.models_dir = models
        self.data_path = data
        self.camera_index = camera

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._jpeg: bytes | None = None
        self.letter = "-"
        self.confidence = 0.0
        self.fps = 0.0
        self.status = "Stopped"
        self.error: str | None = None

        self.deb = LetterDebouncer(window=10, min_votes=5, min_confidence=0.60)

        # Recording state.
        self.recording: str | None = None
        self.remaining = 0
        self._bursts: list[tuple[str, int]] = []

        # Training state.
        self.training = False
        self.last_result: dict | None = None

        X, y = load_samples(self.data_path)
        self.X: list[np.ndarray] = list(X)
        self.y: list[str] = list(y)

        self.model = None
        self.classes: list[str] = []
        self._load_model()

    def _load_model(self) -> None:
        path = self.models_dir / "model.joblib"
        if path.exists():
            self.model = joblib.load(path)
            self.classes = [str(c) for c in self.model.classes_]

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, window: int, min_votes: int, min_confidence: float) -> None:
        if self.running:
            return
        with self._lock:
            self.deb.configure(window, min_votes, min_confidence)
            self.error = None
            self.status = "Starting"
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        with self._lock:
            self.status = "Stopped"
            self.letter, self.confidence, self.fps = "-", 0.0, 0.0
            self.recording, self.remaining = None, 0

    # -- camera thread -----------------------------------------------------

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            with self._lock:
                self.error = (
                    f"Could not open camera {self.camera_index}. On macOS, allow "
                    "camera access for your terminal in System Settings, "
                    "Privacy & Security, Camera."
                )
                self.status = "No camera"
            return

        with self._lock:
            self.status = "Running"
        times: deque[float] = deque(maxlen=30)

        try:
            with HandDetector() as det:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        with self._lock:
                            self.error = "Lost the camera feed."
                        break
                    t0 = time.perf_counter()

                    frame = cv2.flip(frame, 1)  # selfie view
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hit = det.detect(rgb, frame_is_mirrored=True)

                    letter, conf = "-", 0.0
                    if hit is not None:
                        draw_landmarks(frame, hit.raw)
                        if self.recording is not None:
                            self._capture(hit.features)
                        elif self.model is not None:
                            probs = self.model.predict_proba(hit.features[None, :])[0]
                            k = int(probs.argmax())
                            letter, conf = self.classes[k], float(probs[k])
                            self.deb.update(letter, conf)
                    elif self.model is not None and self.recording is None:
                        self.deb.update(NO_HAND, 0.0)

                    self._annotate(frame, letter, conf)
                    ok, buf = cv2.imencode(".jpg", frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    times.append(time.perf_counter() - t0)

                    with self._lock:
                        if ok:
                            self._jpeg = buf.tobytes()
                        self.letter, self.confidence = letter, conf
                        self.fps = 1.0 / max(float(np.mean(times)), 1e-6)
        finally:
            cap.release()
            with self._lock:
                if self.status == "Running":
                    self.status = "Stopped"

    def _capture(self, feats: np.ndarray) -> None:
        """Store one training frame. Only frames with a hand count down."""
        with self._lock:
            self.X.append(feats)
            self.y.append(self.recording)
            self.remaining -= 1
            if self.remaining <= 0:
                label, done = self.recording, self._burst_size
                self._bursts.append((label, done))
                self.recording = None
                save_samples(self.data_path, np.stack(self.X), self.y)

    def _annotate(self, frame: np.ndarray, letter: str, conf: float) -> None:
        h, w = frame.shape[:2]
        if self.recording is not None:
            cv2.rectangle(frame, (0, 0), (w, 44), (24, 24, 160), -1)
            cv2.putText(frame, f"Recording {self.recording}  {self.remaining} left",
                        (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return
        cv2.putText(frame, f"{letter}  {conf:.0%}", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 190, 255), 2)
        cand, frac = self.deb.progress()
        if cand:
            width = int(min(frac, 1.0) * (w - 28))
            cv2.rectangle(frame, (14, h - 24), (14 + width, h - 14),
                          (80, 220, 100), -1)

    # -- recording and training -------------------------------------------

    def record(self, label: str, frames: int) -> None:
        if not self.running:
            raise ValueError("Start the camera before recording.")
        if label not in TEACHABLE:
            raise ValueError(f"{label!r} is not a letter this model can learn.")
        with self._lock:
            self.recording = label
            self.remaining = frames
            self._burst_size = frames

    def undo(self) -> str | None:
        """Drop the most recent burst."""
        with self._lock:
            if not self._bursts:
                return None
            label, n = self._bursts.pop()
            n = min(n, len(self.X))
            del self.X[-n:]
            del self.y[-n:]
            if self.X:
                save_samples(self.data_path, np.stack(self.X), self.y)
            elif self.data_path.exists():
                self.data_path.unlink()
            return label

    def train(self) -> None:
        if self.training:
            return
        self.training = True
        threading.Thread(target=self._train_now, daemon=True).start()

    def _train_now(self) -> None:
        try:
            with self._lock:
                X = np.stack(self.X) if self.X else np.empty((0, 1))
                y = list(self.y)
            result = train_model(X, y, self.models_dir)
            self._load_model()
            with self._lock:
                self.last_result = result
                self.error = None
        except Exception as exc:  # surfaced in the page, not the console
            with self._lock:
                self.last_result = None
                self.error = f"Training failed: {exc}"
        finally:
            self.training = False

    # -- readers -----------------------------------------------------------

    def frames_stream(self):
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with self._lock:
                jpeg = self._jpeg
            if jpeg is not None:
                yield boundary + jpeg + b"\r\n"
            time.sleep(0.04)

    def snapshot(self) -> dict:
        with self._lock:
            per_label = counts(self.y)
            return {
                "letter": self.letter,
                "confidence": round(self.confidence, 3),
                "fps": round(self.fps, 1),
                "status": self.status,
                "running": self.running,
                "text": self.deb.text,
                "error": self.error,
                "has_model": self.model is not None,
                "recording": self.recording,
                "remaining": max(self.remaining, 0),
                "training": self.training,
                "can_undo": bool(self._bursts),
                "total_samples": len(self.X),
                "counts": {c: per_label.get(c, 0) for c in TEACHABLE},
                "recommended": RECOMMENDED_PER_CLASS,
                "result": self.last_result,
            }


engine: Engine | None = None


@app.route("/")
def index():
    return render_template("index.html", teachable=TEACHABLE)


@app.route("/video")
def video():
    return Response(engine.frames_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def state():
    return jsonify(engine.snapshot())


def _ok():
    return jsonify(engine.snapshot())


def _bad(exc: Exception):
    # The spread must come first: snapshot() carries its own "error" key, so
    # spreading it afterwards would overwrite the message with None.
    return jsonify({**engine.snapshot(), "error": str(exc)}), 400


@app.route("/api/start", methods=["POST"])
def start():
    body = request.get_json(silent=True) or {}
    try:
        window = int(body.get("window", 10))
        votes = min(int(body.get("min_votes", 5)), window)
        engine.start(window, votes, float(body.get("confidence", 0.60)))
    except (TypeError, ValueError) as exc:
        return _bad(exc)
    return _ok()


@app.route("/api/stop", methods=["POST"])
def stop():
    engine.stop()
    return _ok()


@app.route("/api/record", methods=["POST"])
def record():
    body = request.get_json(silent=True) or {}
    try:
        frames = max(5, min(int(body.get("frames", 30)), 200))
        engine.record(str(body.get("letter", "")), frames)
    except (TypeError, ValueError) as exc:
        return _bad(exc)
    return _ok()


@app.route("/api/undo", methods=["POST"])
def undo():
    engine.undo()
    return _ok()


@app.route("/api/train", methods=["POST"])
def train():
    engine.train()
    return _ok()


@app.route("/api/space", methods=["POST"])
def space():
    engine.deb.insert(SPACE)
    return _ok()


@app.route("/api/backspace", methods=["POST"])
def backspace():
    engine.deb.insert(DELETE)
    return _ok()


@app.route("/api/clear", methods=["POST"])
def clear():
    engine.deb.reset()
    return _ok()


def main() -> None:
    global engine
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--data", type=Path, default=Path("data/samples.npz"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    engine = Engine(args.models, args.data, args.camera)
    print(f"\n  open http://{args.host}:{args.port}\n")
    # The reloader would open the camera twice, on the same device.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
