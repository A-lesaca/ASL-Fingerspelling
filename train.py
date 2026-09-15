"""Train the ASL letter classifier on a landmark feature set.

Consumes the .npz written by convert.py or sample.py and writes:

    models/model.joblib   a fitted scikit-learn classifier
    models/classes.json   the label order, for inspection

Note that the label order is read back from the fitted estimator's
``classes_`` attribute at inference time, not from classes.json. Deriving it
from the model itself makes it impossible for the two to drift apart -- a
stale classes.json would otherwise produce confident, wrong letters with no
error anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import VECTOR_DIM


def load_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate one or more .npz feature files."""
    Xs, ys = [], []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"dataset not found: {p}")
        d = np.load(p, allow_pickle=True)
        Xs.append(d["X"].astype(np.float32))
        ys.append(d["y"].astype(str))
        print(f"{p}: {len(d['X'])} samples")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    if X.ndim != 2 or X.shape[1] != VECTOR_DIM:
        raise SystemExit(
            f"expected (N, {VECTOR_DIM}) features, got {X.shape}. "
            "Datasets built before the feature change need rebuilding."
        )
    return X, y


def build_estimator(kind: str, seed: int):
    """Construct one of three classifiers over the same feature vector."""
    if kind == "forest":
        # The reference project's choice. Handles unscaled, mixed-unit features
        # (coordinates, radians, ratios) without any preprocessing, because
        # trees split on thresholds rather than on distances.
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "svm":
        # Distance-based, so it does need scaling. Often the stronger model on
        # small landmark datasets, at the cost of slower prediction.
        return make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", probability=True,
                class_weight="balanced", random_state=seed),
        )
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000,
                          early_stopping=True, random_state=seed),
        )
    raise SystemExit(f"unknown classifier: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, nargs="+", required=True,
                    help="one or more .npz files from convert.py / sample.py")
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--classifier", choices=["forest", "svm", "mlp"],
                    default="forest")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cross-validate", action="store_true",
                    help="also run 5-fold CV on the full set (slower)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y = load_dataset(args.data)

    labels, counts = np.unique(y, return_counts=True)
    print(f"\n{len(X)} samples, {len(labels)} classes, {X.shape[1]} features")
    thin = [(l, int(n)) for l, n in zip(labels, counts) if n < 40]
    if thin:
        print("under-represented: " + ", ".join(f"{l}={n}" for l, n in thin))
    if counts.min() < 2:
        raise SystemExit("every class needs at least 2 samples to stratify")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, shuffle=True,
        stratify=y, random_state=args.seed,
    )
    print(f"train {len(X_tr)}, test {len(X_te)}")

    model = build_estimator(args.classifier, args.seed)
    print(f"\nfitting {args.classifier}...")
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    accuracy = float((y_pred == y_te).mean())
    print(f"\nheld-out accuracy: {accuracy:.1%}\n")
    print(classification_report(y_te, y_pred, zero_division=0))

    if args.cross_validate:
        scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
        print(f"5-fold CV: {scores.mean():.1%} +/- {scores.std():.1%}")

    # Overall accuracy hides the pairs that actually break transcription.
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    pairs = [(labels[i], labels[j], int(cm[i, j]))
             for i, j in zip(*np.nonzero(cm))]
    if pairs:
        print("top confusions (true -> predicted):")
        for t, p, n in sorted(pairs, key=lambda r: -r[2])[:8]:
            print(f"  {t} -> {p}: {n}")

    args.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.models / "model.joblib")
    (args.models / "classes.json").write_text(json.dumps(sorted(labels.tolist())))
    print(f"\nwrote {args.models / 'model.joblib'} and "
          f"{args.models / 'classes.json'} ({len(labels)} classes)")


if __name__ == "__main__":
    main()