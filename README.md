# ASL Fingerspelling

A browser-based ASL fingerspelling trainer. Record your own hand signs with a
webcam, train a classifier on them, then get live letter-by-letter
transcription and practice drills.

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

## Resetting saved data

To wipe recorded samples, the trained model, custom gestures, and (optionally)
practice history:

```bash
python reset.py            # asks for confirmation
python reset.py --yes      # skip confirmation
python reset.py --keep-history
python reset.py --port     # frees port 5000
```
