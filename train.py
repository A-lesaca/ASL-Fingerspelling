"""Train the classifier and report both accuracy numbers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42


def build_model(input_dim: int, n_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def jitter(X: np.ndarray, y: np.ndarray, factor: int, sigma: float, rng) -> tuple:
    """Augment with small Gaussian noise on the landmarks.

    Translation, scale and rotation are already normalised away, so the only
    variation left to simulate is landmark jitter -- MediaPipe placing a joint
    a pixel or two off. Keep sigma small; large noise makes similar letters
    (M/N/S/T) collide and actively hurts.
    """
    if factor <= 1:
        return X, y
    noisy = [X] + [X + rng.normal(0, sigma, X.shape).astype(np.float32)
                   for _ in range(factor - 1)]
    return np.concatenate(noisy), np.tile(y, factor)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, default=None,
                    help="your own recorded samples -- the honest test set")
    ap.add_argument("--outdir", type=Path, default=Path("models"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--augment", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.01)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    tf.keras.utils.set_random_seed(SEED)

    d = np.load(args.train, allow_pickle=True)
    X, y_raw = d["X"].astype(np.float32), d["y"]

    classes = sorted(set(y_raw.tolist()))
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[c] for c in y_raw])
    print(f"{len(X)} samples, {len(classes)} classes: {' '.join(classes)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_tr, y_tr = jitter(X_tr, y_tr, args.augment, args.noise, rng)

    model = build_model(X.shape[1], len(classes))
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=128,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
            ),
        ],
        verbose=2,
    )

    in_dist = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"\n  in-distribution accuracy (public split) : {in_dist:.4f}")

    honest = None
    if args.holdout and args.holdout.exists():
        h = np.load(args.holdout, allow_pickle=True)
        keep = np.isin(h["y"], classes)
        Xh = h["X"].astype(np.float32)[keep]
        yh = np.array([index[c] for c in h["y"][keep]])

        honest = model.evaluate(Xh, yh, verbose=0)[1]
        print(f"  held-out accuracy (my own webcam)      : {honest:.4f}")
        print(f"  generalisation gap                     : {in_dist - honest:.4f}\n")

        pred = model.predict(Xh, verbose=0).argmax(axis=1)
        print(classification_report(yh, pred, target_names=classes, zero_division=0))

        cm = confusion_matrix(yh, pred, labels=range(len(classes)))
        pairs = [
            (cm[i, j], classes[i], classes[j])
            for i in range(len(classes)) for j in range(len(classes)) if i != j
        ]
        print("most confused pairs:")
        for n, a, b in sorted(pairs, reverse=True)[:6]:
            if n:
                print(f"  {a} mistaken for {b}: {n}")
    else:
        print("  no held-out set given -- the number above is NOT a real"
              " generalisation estimate.\n")

    args.outdir.mkdir(parents=True, exist_ok=True)
    model.save(args.outdir / "model.keras")
    (args.outdir / "classes.json").write_text(json.dumps(classes))
    (args.outdir / "metrics.json").write_text(
        json.dumps({"in_distribution": float(in_dist),
                    "held_out": float(honest) if honest is not None else None},
                   indent=2)
    )
    print(f"saved -> {args.outdir}")
    print("\nfor the browser demo:  tensorflowjs_converter --input_format=keras "
          f"{args.outdir / 'model.keras'} web/model")


if __name__ == "__main__":
    main()