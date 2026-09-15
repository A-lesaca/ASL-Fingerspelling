"""Dataset storage and model fitting, shared by the web app.

Kept separate from app.py so the training logic can be tested without a
camera, a browser or a running server.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from features import VECTOR_DIM

MIN_PER_CLASS = 2          # below this a class cannot be split
RECOMMENDED_PER_CLASS = 40  # below this the UI warns


def load_samples(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load the sample store, or return an empty one."""
    if not path.exists():
        return np.empty((0, VECTOR_DIM), dtype=np.float32), []
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    if X.size and X.shape[1] != VECTOR_DIM:
        raise ValueError(
            f"{path} holds {X.shape[1]}-wide rows but this version builds "
            f"{VECTOR_DIM}. Delete it and record again."
        )
    return X, [str(v) for v in d["y"]]


def save_samples(path: Path, X: np.ndarray, y: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=np.array(y))


def counts(y: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label in y:
        out[label] = out.get(label, 0) + 1
    return out


def train_model(
    X: np.ndarray,
    y: list[str],
    models_dir: Path,
    seed: int = 0,
) -> dict:
    """Fit a classifier and write it to disk. Returns a result summary."""
    labels, label_counts = np.unique(np.array(y), return_counts=True)

    if len(labels) < 2:
        raise ValueError("Record at least two different letters before training.")
    too_thin = [str(l) for l, n in zip(labels, label_counts) if n < MIN_PER_CLASS]
    if too_thin:
        raise ValueError(
            "These letters have too few samples to train on: "
            + ", ".join(too_thin)
        )

    y_arr = np.array(y)
    # Stratifying needs at least one sample per class on each side, which a
    # 20% split cannot guarantee for a class with only a handful of samples.
    test_size = 0.2 if label_counts.min() >= 5 else 0.0

    if test_size:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_arr, test_size=test_size, shuffle=True,
            stratify=y_arr, random_state=seed,
        )
    else:
        X_tr, y_tr, X_te, y_te = X, y_arr, None, None

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)

    accuracy = None
    confusions: list[tuple[str, str, int]] = []
    if X_te is not None and len(X_te):
        pred = model.predict(X_te)
        accuracy = float((pred == y_te).mean())
        cm = confusion_matrix(y_te, pred, labels=labels)
        np.fill_diagonal(cm, 0)
        confusions = sorted(
            ((str(labels[i]), str(labels[j]), int(cm[i, j]))
             for i, j in zip(*np.nonzero(cm))),
            key=lambda r: -r[2],
        )[:5]

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "model.joblib")
    (models_dir / "classes.json").write_text(
        json.dumps(sorted(str(l) for l in labels))
    )

    return {
        "accuracy": accuracy,
        "samples": int(len(X)),
        "classes": [str(l) for l in labels],
        "confusions": confusions,
        "thin": [str(l) for l, n in zip(labels, label_counts)
                 if n < RECOMMENDED_PER_CLASS],
    }
