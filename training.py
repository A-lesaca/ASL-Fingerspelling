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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from features import VECTOR_DIM

MIN_PER_CLASS = 2          # below this a class cannot be split
RECOMMENDED_PER_CLASS = 40  # below this the UI warns


def load_samples(path: Path) -> tuple[np.ndarray, list[str], list[int]]:
    """Load the sample store, or return an empty one.

    The third value is the burst each sample came from. Frames recorded in one
    burst are near-duplicates of each other, so which burst a sample belongs to
    matters when splitting -- see train_model.
    """
    if not path.exists():
        return np.empty((0, VECTOR_DIM), dtype=np.float32), [], []
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    if X.size and X.shape[1] != VECTOR_DIM:
        raise ValueError(
            f"{path} holds {X.shape[1]}-wide rows but this version builds "
            f"{VECTOR_DIM}. Delete it and record again."
        )
    y = [str(v) for v in d["y"]]
    groups = [int(g) for g in d["groups"]] if "groups" in d else list(range(len(y)))
    return X, y, groups


def save_samples(path: Path, X: np.ndarray, y: list[str],
                 groups: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if groups is None:
        groups = list(range(len(y)))
    np.savez_compressed(path, X=X, y=np.array(y),
                        groups=np.array(groups, dtype=np.int64))


def counts(y: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label in y:
        out[label] = out.get(label, 0) + 1
    return out


def _split(X, y_arr, groups, seed):
    """Hold out whole bursts where possible, individual frames otherwise.

    A burst is one key press: thirty frames of the same hand barely moving, so
    they are near-duplicates. Splitting those at random puts copies of the same
    moment on both sides, and the reported accuracy then measures memorisation
    rather than recognition -- comfortably over-optimistic. Holding out whole
    bursts gives a number that means something, but it needs at least two
    bursts of every letter.
    """
    per_class_groups = {}
    for label, g in zip(y_arr, groups):
        per_class_groups.setdefault(label, set()).add(g)
    fewest = min((len(gs) for gs in per_class_groups.values()), default=0)

    if fewest >= 2:
        splitter = StratifiedGroupKFold(n_splits=min(5, fewest), shuffle=True,
                                        random_state=seed)
        tr, te = next(splitter.split(X, y_arr, groups=np.array(groups)))
        return X[tr], X[te], y_arr[tr], y_arr[te], True

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_arr, test_size=0.2, shuffle=True, stratify=y_arr,
        random_state=seed)
    return X_tr, X_te, y_tr, y_te, False


def train_model(
    X: np.ndarray,
    y: list[str],
    models_dir: Path,
    seed: int = 0,
    groups: list[int] | None = None,
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
    if groups is None:
        groups = list(range(len(y)))

    honest_split = False
    if label_counts.min() >= 5:
        X_tr, X_te, y_tr, y_te, honest_split = _split(X, y_arr, groups, seed)
    else:
        # Too few samples to hold anything back at all.
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
    per_class: list[tuple[str, float]] = []
    if X_te is not None and len(X_te):
        pred = model.predict(X_te)
        accuracy = float((pred == y_te).mean())
        cm = confusion_matrix(y_te, pred, labels=labels)

        # Recall per label: of the times this letter was signed, how often was
        # it read correctly. Overall accuracy hides a letter that is always
        # wrong when the other twenty are always right.
        totals = cm.sum(axis=1)
        per_class = sorted(
            ((str(labels[i]), float(cm[i, i] / totals[i]) if totals[i] else 0.0)
             for i in range(len(labels))),
            key=lambda r: r[1],
        )

        np.fill_diagonal(cm, 0)
        confusions = sorted(
            ((str(labels[i]), str(labels[j]), int(cm[i, j]))
             for i, j in zip(*np.nonzero(cm))),
            key=lambda r: -r[2],
        )[:6]

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
        "per_class": per_class,
        "honest_split": honest_split,
        "bursts": {label: len({g for lab, g in zip(y, groups) if lab == label})
                   for label in (str(l) for l in labels)},
        "per_label_counts": {str(l): int(n) for l, n in zip(labels, label_counts)},
        "thin": [str(l) for l, n in zip(labels, label_counts)
                 if n < RECOMMENDED_PER_CLASS],
    }