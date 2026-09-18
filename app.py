from __future__ import annotations

import argparse
import json
import re
import string
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import (Flask, Response, jsonify, render_template, request,
                   send_file)

from detector import HandDetector, draw_landmarks
from handshapes import describe
from practice import PracticeSession
from smoothing import DELETE, NO_HAND, SPACE, LetterDebouncer
from training import (RECOMMENDED_PER_CLASS, counts, load_samples,
                      save_samples, train_model)

# J and Z are drawn with motion, which a single-frame classifier cannot
# represent, so they are left out of the alphabet everywhere.
MOTION_LETTERS = {"J", "Z"}
ALPHABET = [c for c in string.ascii_uppercase if c not in MOTION_LETTERS]

BUILT_IN = list(ALPHABET)

# Custom gestures are ordinary labels

GESTURE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 '-]{0,23}$")

app = Flask(__name__)


class Engine:
    """Owns the camera thread and every piece of state it touches."""

    def __init__(self, models: Path, data: Path, camera: int,
                 window: int = 10, min_votes: int = 5,
                 confidence: float = 0.60) -> None:
        self.defaults = (window, min(min_votes, window), confidence)
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

        self.deb = LetterDebouncer(window=window, min_votes=min(min_votes, window),
                                   min_confidence=confidence)

        # Recording state. `pending` is a letter waiting out its countdown:
        self.recording: str | None = None
        self.remaining = 0
        self.pending: str | None = None
        self.countdown_ends = 0.0
        self.countdown = 3.0
        self._bursts: list[tuple[str, int]] = []

        # Practice state.
        self.practice: PracticeSession | None = None
        self.last_practice: dict | None = None
        self.history_path = data.with_name("practice.json")

        # Training state.
        self.training = False
        self.last_result: dict | None = None

        self.gestures_path = data.with_name("gestures.json")
        self.thumb_dir = data.with_name("thumbs")
        self.gestures: list[str] = self._load_gestures()

        X, y, groups = load_samples(self.data_path)
        self.X: list[np.ndarray] = list(X)
        self.y: list[str] = list(y)
        self.groups: list[int] = list(groups)
        self._next_group = (max(self.groups) + 1) if self.groups else 0

        self.model = None
        self.classes: list[str] = []
        self._load_model()

    @staticmethod
    def thumb_name(label: str) -> str:
        """A filename-safe stem. Gesture names may contain spaces and quotes."""
        return "".join(ch if ch.isalnum() else "_" for ch in label) or "_"

    def thumb_path(self, label: str) -> Path:
        return self.thumb_dir / f"{self.thumb_name(label)}.jpg"

    def _save_thumb(self, frame: np.ndarray, raw: list, label: str) -> None:
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in raw]
        ys = [lm.y * h for lm in raw]
        if not xs or not ys:
            return
        pad = 0.35 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        x0, x1 = int(max(min(xs) - pad, 0)), int(min(max(xs) + pad, w))
        y0, y1 = int(max(min(ys) - pad, 0)), int(min(max(ys) + pad, h))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return
        crop = frame[y0:y1, x0:x1]
        side = max(crop.shape[:2])
        square = np.full((side, side, 3), 24, np.uint8)
        oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
        square[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.thumb_path(label)),
                    cv2.resize(square, (128, 128), interpolation=cv2.INTER_AREA),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82])

    def _load_gestures(self) -> list[str]:
        if not self.gestures_path.exists():
            return []
        try:
            raw = json.loads(self.gestures_path.read_text())
            return [str(g) for g in raw if isinstance(g, str)]
        except (ValueError, OSError):
            return []  

    def _save_gestures(self) -> None:
        self.gestures_path.parent.mkdir(parents=True, exist_ok=True)
        self.gestures_path.write_text(json.dumps(self.gestures))

    @property
    def teachable(self) -> list[str]:
        return BUILT_IN + self.gestures

    def add_gesture(self, name: str) -> str:
        name = " ".join(name.split()).upper()
        if not name:
            raise ValueError("Give the gesture a name.")
        if not GESTURE_PATTERN.match(name):
            raise ValueError(
                "Use letters, digits, spaces, apostrophes or hyphens (max 24)."
            )
        if len(name) == 1:
            raise ValueError("Single letters are already covered by the alphabet.")
        if name in self.teachable:
            raise ValueError(f"{name} already exists.")
        with self._lock:
            self.gestures.append(name)
            self._save_gestures()
        return name

    def remove_gesture(self, name: str) -> None:
        with self._lock:
            if name not in self.gestures:
                raise ValueError(f"No gesture called {name}.")
            self.gestures.remove(name)
            keep = [i for i, label in enumerate(self.y) if label != name]
            self.X = [self.X[i] for i in keep]
            self.y = [self.y[i] for i in keep]
            self.groups = [self.groups[i] for i in keep]
            self._bursts = [b for b in self._bursts if b[0] != name]
            self.thumb_path(name).unlink(missing_ok=True)
            self._save_gestures()
            if self.X:
                save_samples(self.data_path, np.stack(self.X), self.y, self.groups)
            elif self.data_path.exists():
                self.data_path.unlink()

    def _load_model(self) -> None:
        path = self.models_dir / "model.joblib"
        if path.exists():
            self.model = joblib.load(path)
            self.classes = [str(c) for c in self.model.classes_]

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        with self._lock:
            self.deb.configure(*self.defaults)
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
            self.recording, self.remaining, self.pending = None, 0, None

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

                    self._tick_countdown()
                    frame = cv2.flip(frame, 1)  # selfie view
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hit = det.detect(rgb, frame_is_mirrored=True)

                    letter, conf = "-", 0.0
                    if hit is not None:
                        draw_landmarks(frame, hit.raw)
                        if self.recording is not None:
                            label = self.recording
                            first = self.remaining == self._burst_size
                            self._capture(hit.features)
                            if first:
                                self._save_thumb(frame, hit.raw, label)
                        elif self.model is not None:
                            probs = self.model.predict_proba(hit.features[None, :])[0]
                            k = int(probs.argmax())
                            letter, conf = self.classes[k], float(probs[k])
                            committed = self.deb.update(letter, conf)
                            if committed is not None:
                                self._score(committed)
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

    def _tick_countdown(self) -> None:
        """Start a scheduled burst once its countdown has run out."""
        if self.pending is None:
            return
        if time.time() >= self.countdown_ends:
            with self._lock:
                self.recording = self.pending
                self.remaining = self._burst_size
                self.pending = None

    def _capture(self, feats: np.ndarray) -> None:
        """Store one training frame. Only frames with a hand count down."""
        with self._lock:
            self.X.append(feats)
            self.y.append(self.recording)
            self.groups.append(self._group_id)
            self.remaining -= 1
            if self.remaining <= 0:
                label, done = self.recording, self._burst_size
                self._bursts.append((label, done))
                self.recording = None
                save_samples(self.data_path, np.stack(self.X), self.y, self.groups)

    def _score(self, committed: str) -> None:
        """Feed a committed letter to the running practice session, if any."""
        session = self.practice
        if session is None or session.done:
            return
        session.observe(committed)
        if session.done:
            self._finish_practice()

    def start_practice(self, word: str) -> None:
        if self.model is None:
            raise ValueError("Train a model before practising.")
        session = PracticeSession(word)
        unknown = sorted({c for c in session.target if c != " "} - set(self.classes))
        if unknown:
            raise ValueError(
                "Not trained on: " + ", ".join(unknown) + ". Record those first."
            )
        if " " in session.target and SPACE not in self.classes:
            raise ValueError("Record a 'space' sign before practising phrases.")
        with self._lock:
            self.practice = session
            self.last_practice = None
            self.deb.reset()

    def stop_practice(self) -> None:
        with self._lock:
            if self.practice is not None:
                self.last_practice = self.practice.summary()
                self._append_history(self.last_practice)
            self.practice = None

    def skip_letter(self) -> None:
        with self._lock:
            if self.practice is None:
                raise ValueError("No practice run in progress.")
            self.practice.skip()
            if self.practice.done:
                self._finish_practice_locked()

    def _finish_practice(self) -> None:
        with self._lock:
            self._finish_practice_locked()

    def _finish_practice_locked(self) -> None:
        if self.practice is None:
            return
        self.last_practice = self.practice.summary()
        self._append_history(self.last_practice)
        self.practice = None

    def _append_history(self, summary: dict) -> None:
        """Keep a rolling log of runs, so results survive a restart.

        The page can show the last 100 runs, and the user can download them.
        """
        record = {
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": summary["target"],
            "done": summary["done"],
            "first_try_rate": round(summary["first_try_rate"], 3),
            "accuracy": round(summary["accuracy"], 3),
            "seconds": summary["seconds"],
            "seconds_per_letter": summary["seconds_per_letter"],
            "confusions": summary["confusions"],
        }
        try:
            history = json.loads(self.history_path.read_text()) \
                if self.history_path.exists() else []
        except (ValueError, OSError):
            history = []
        history.append(record)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(history[-100:], indent=1))

    def history(self) -> list[dict]:
        try:
            return json.loads(self.history_path.read_text()) \
                if self.history_path.exists() else []
        except (ValueError, OSError):
            return []

    def _annotate(self, frame: np.ndarray, letter: str, conf: float) -> None:
        """Draw the whole interface into the frame.

        The top band shows what has been typed so far, or the recording countdown.
        The bottom left shows the current letter, with the vote progress beneath it.
        The bottom right shows what the keys do.
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Top band: what has been typed so far, or the recording countdown.
        cv2.rectangle(frame, (0, 0), (w, 46), (26, 26, 24), -1)
        if self.pending is not None:
            left = max(0.0, self.countdown_ends - time.time())
            cv2.rectangle(frame, (0, 0), (w, 46), (20, 120, 190), -1)
            cv2.putText(frame, f"Get ready: {self.pending}   {left:.0f}",
                        (14, 32), font, 0.9, (255, 255, 255), 2)
            guide = describe(self.pending)
            cv2.rectangle(frame, (0, 46), (w, 74), (20, 90, 140), -1)
            cv2.putText(frame, guide[:70], (14, 65), font, 0.5,
                        (235, 240, 245), 1)
        elif self.recording is not None:
            cv2.rectangle(frame, (0, 0), (w, 46), (30, 30, 170), -1)
            cv2.putText(frame, f"REC {self.recording}  {self.remaining} left",
                        (14, 32), font, 0.9, (255, 255, 255), 2)
        elif self.training:
            cv2.putText(frame, "training...", (14, 32), font, 0.9,
                        (80, 200, 255), 2)
        elif self.practice is not None:
            done, cur = self.practice.typed(), self.practice.current or ""
            cv2.putText(frame, done, (14, 33), font, 0.95, (120, 240, 120), 2)
            (dw, _), _ = cv2.getTextSize(done, font, 0.95, 2)
            cv2.putText(frame, cur, (14 + dw, 33), font, 0.95, (80, 200, 255), 2)
            (cw, _), _ = cv2.getTextSize(cur, font, 0.95, 2)
            rest = self.practice.target[len(done) + len(cur):]
            cv2.putText(frame, rest, (14 + dw + cw, 33), font, 0.95, (130, 130, 126), 2)
            cv2.putText(frame, f"{self.practice.elapsed:4.1f}s", (w - 96, 30),
                        font, 0.6, (150, 150, 145), 1)
        else:
            shown = self.deb.text[-30:] or "..."
            cv2.putText(frame, shown, (14, 33), font, 0.95, (255, 255, 255), 2)
            cv2.putText(frame, f"{self.fps:.0f} fps", (w - 96, 30), font,
                        0.6, (150, 150, 145), 1)

        if self.recording is not None or self.pending is not None:
            return

        # Bottom left: the current letter, with the vote progress beneath it.
        box, pad = 74, 14
        top = h - box - pad - 12
        cv2.rectangle(frame, (pad, top), (pad + box, top + box), (26, 26, 24), -1)
        if self.model is None:
            cv2.putText(frame, "?", (pad + 22, top + 56), font, 1.6,
                        (120, 120, 115), 3)
        else:
            cv2.putText(frame, letter if letter != "-" else "-",
                        (pad + 18, top + 56), font, 1.6, (120, 240, 120), 3)
        cand, frac = self.deb.progress()
        bar = int(min(frac, 1.0) * box) if cand else 0
        cv2.rectangle(frame, (pad, top + box + 4), (pad + box, top + box + 12),
                      (40, 40, 38), -1)
        if bar:
            cv2.rectangle(frame, (pad, top + box + 4), (pad + bar, top + box + 12),
                          (120, 240, 120), -1)

        # Bottom right: what the keys do.
        hints = ["letter key: record", "enter: train", "esc: clear"]
        if self.model is None:
            hints = ["press a letter to record it", "then enter to train"]
        for i, line in enumerate(hints):
            y = h - 14 - (len(hints) - 1 - i) * 18
            (tw, _), _ = cv2.getTextSize(line, font, 0.5, 1)
            cv2.putText(frame, line, (w - tw - 14, y), font, 0.5,
                        (170, 170, 164), 1)

    def record(self, label: str, frames: int) -> None:
        if not self.running:
            raise ValueError("Start the camera before recording.")
        if label not in self.teachable:
            raise ValueError(f"{label!r} is not a letter this model can learn.")
        with self._lock:
            self._burst_size = frames
            self._group_id = self._next_group
            self._next_group += 1
            self.pending = label
            self.countdown_ends = time.time() + self.countdown
            self.error = None

    def undo(self) -> str | None:
        """Drop the most recent burst."""
        with self._lock:
            if not self._bursts:
                return None
            label, n = self._bursts.pop()
            n = min(n, len(self.X))
            del self.X[-n:]
            del self.y[-n:]
            del self.groups[-n:]
            if self.X:
                save_samples(self.data_path, np.stack(self.X), self.y, self.groups)
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
                groups = list(self.groups)
            result = train_model(X, y, self.models_dir, groups=groups)
            self._load_model()
            with self._lock:
                self.last_result = result
                self.error = None
        except Exception as exc:  
            with self._lock:
                self.last_result = None
                self.error = f"Training failed: {exc}"
        finally:
            self.training = False

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
                "pending": self.pending,
                "countdown": max(0.0, round(self.countdown_ends - time.time(), 1))
                             if self.pending else 0.0,
                "remaining": max(self.remaining, 0),
                "training": self.training,
                "can_undo": bool(self._bursts),
                "total_samples": len(self.X),
                "counts": {c: per_label.get(c, 0) for c in self.teachable},
                "gestures": list(self.gestures),
                "thumbs": [c for c in self.teachable
                           if self.thumb_path(c).exists()],
                "hints": {c: describe(c) for c in self.teachable},
                "recommended": RECOMMENDED_PER_CLASS,
                "result": self.last_result,
                "practice": ({
                    "target": self.practice.target,
                    "typed": self.practice.typed(),
                    "current": self.practice.current,
                    "hint": describe(self.practice.current or ""),
                    "elapsed": round(self.practice.elapsed, 1),
                    "summary": self.practice.summary(),
                } if self.practice is not None else None),
                "last_practice": self.last_practice,
            }


engine: Engine | None = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(engine.frames_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/thumb/<path:label>")
def thumb(label: str):
    path = engine.thumb_path(label)
    if not path.exists():
        return ("", 404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/state")
def state():
    return jsonify(engine.snapshot())


def _ok():
    return jsonify(engine.snapshot())


def _bad(exc: Exception):
    engine.error = str(exc)
    return jsonify({**engine.snapshot(), "error": str(exc)}), 400


@app.route("/api/start", methods=["POST"])
def start():
    engine.start()
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


@app.route("/api/gesture", methods=["POST"])
def gesture():
    body = request.get_json(silent=True) or {}
    try:
        engine.add_gesture(str(body.get("name", "")))
    except (TypeError, ValueError) as exc:
        return _bad(exc)
    return _ok()


@app.route("/api/gesture/remove", methods=["POST"])
def gesture_remove():
    body = request.get_json(silent=True) or {}
    try:
        engine.remove_gesture(str(body.get("name", "")))
    except (TypeError, ValueError) as exc:
        return _bad(exc)
    return _ok()


@app.route("/api/practice/start", methods=["POST"])
def practice_start():
    body = request.get_json(silent=True) or {}
    try:
        engine.start_practice(str(body.get("word", "")))
    except (TypeError, ValueError) as exc:
        return _bad(exc)
    return _ok()


@app.route("/api/practice/stop", methods=["POST"])
def practice_stop():
    engine.stop_practice()
    return _ok()


@app.route("/api/practice/skip", methods=["POST"])
def practice_skip():
    try:
        engine.skip_letter()
    except (TypeError, ValueError) as exc:
        return _bad(exc)
    return _ok()


@app.route("/api/practice/history")
def practice_history():
    return jsonify(engine.history())


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
    ap.add_argument("--confidence", type=float, default=0.60,
                    help="ignore frames the classifier is less sure than this")
    ap.add_argument("--window", type=int, default=10,
                    help="how many recent frames vote on each letter")
    ap.add_argument("--min-votes", type=int, default=5,
                    help="how many of those frames must agree to commit")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="seconds to get into position after pressing a key")
    args = ap.parse_args()

    engine = Engine(args.models, args.data, args.camera,
                    window=args.window, min_votes=args.min_votes,
                    confidence=args.confidence)
    engine.countdown = max(0.0, args.countdown)
    print(f"\n  open http://{args.host}:{args.port}\n")
    # The reloader would open the camera twice, on the same device.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()