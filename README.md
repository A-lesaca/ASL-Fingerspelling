# ASL Fingerspelling

A browser-based ASL fingerspelling trainer. Record your own hand signs with a
webcam, train a classifier on them, then get live letter-by-letter
transcription and practice drills — all from one page, no separate scripts.

Hand tracking is done with MediaPipe; a classifier is trained on the
landmarks scikit-learn produces. Only static handshapes are supported, so
**J and Z are excluded for this project** (they require motion, which a single-frame
classifier can't capture).

## Features

- **Record** — press a letter key to capture a short burst of webcam frames
  for that handshape.
- **Train** — fit a classifier on your recorded samples, right from the page.
- **Transcribe** — live webcam feed with real-time letter predictions,
  smoothed over multiple frames before committing to text.
- **Custom gestures** — define your own signs (e.g. whole words) beyond the
  alphabet; they train the same way as letters.
- **Practice mode** — type a target word/phrase with your hands and get
  accuracy, timing, and confusion stats, saved to a history log.

## Requirements

- Python 3.10+
- A webcam

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Then open the printed URL (default `http://127.0.0.1:5000`) in a browser.

Useful flags:

| Flag           | Default | Description                                  |
|----------------|---------|-----------------------------------------------|
| `--camera`     | `0`     | Camera index                                  |
| `--host`       | `127.0.0.1` | Host to bind                              |
| `--port`       | `5000`  | Port to bind                                  |
| `--confidence` | `0.60`  | Minimum confidence to accept a prediction     |
| `--window`     | `10`    | Frames considered per letter vote             |
| `--min-votes`  | `5`     | Votes needed to commit a letter               |
| `--countdown`  | `3.0`   | Seconds to get into position before recording |

## Resetting saved data

To wipe recorded samples, the trained model, custom gestures, and (optionally)
practice history:

```bash
python reset.py            # asks for confirmation
python reset.py --yes      # skip confirmation
python reset.py --keep-history
python reset.py --port     # also frees port 5000, in case the app is stuck running
```

## Project structure

```
app.py          # Flask app: camera loop, recording, training, API routes
detector.py     # MediaPipe hand detection + landmark drawing
features.py     # Landmark -> feature vector conversion
training.py     # Model training and sample storage
smoothing.py    # Multi-frame letter debouncing
practice.py     # Practice session scoring/state
handshapes.py   # Text hints describing each handshape
reset.py        # Utility to clear saved data/models
templates/      # Web page
data/, models/  # Saved samples, gestures, trained model
```
