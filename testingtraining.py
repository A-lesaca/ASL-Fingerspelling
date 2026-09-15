"""Tests for dataset storage and model fitting."""

import numpy as np
import pytest

from features import VECTOR_DIM, build_features
from training import counts, load_samples, save_samples, train_model

rng = np.random.default_rng(4)


def sample_set(labels, per_class):
    """Distinct synthetic hand poses, a few noisy samples each."""
    X, y = [], []
    for label in labels:
        proto = rng.uniform(0.3, 0.7, size=(21, 3))
        for _ in range(per_class):
            X.append(build_features(proto + rng.normal(0, 0.006, size=(21, 3))))
            y.append(label)
    return np.stack(X), y


def test_missing_store_loads_as_empty(tmp_path):
    X, y = load_samples(tmp_path / "nothing.npz")
    assert X.shape == (0, VECTOR_DIM)
    assert y == []


def test_round_trips_through_disk(tmp_path):
    X, y = sample_set(["A", "B"], 3)
    path = tmp_path / "data" / "samples.npz"
    save_samples(path, X, y)
    X2, y2 = load_samples(path)
    assert np.allclose(X, X2)
    assert y2 == y


def test_rejects_a_store_from_an_older_feature_version(tmp_path):
    path = tmp_path / "old.npz"
    np.savez_compressed(path, X=np.zeros((4, 63), np.float32),
                        y=np.array(["A"] * 4))
    with pytest.raises(ValueError, match="63"):
        load_samples(path)


def test_counts_tallies_labels():
    assert counts(["A", "B", "A"]) == {"A": 2, "B": 1}


class TestTraining:
    def test_fits_and_writes_both_files(self, tmp_path):
        X, y = sample_set(["A", "B", "C"], 20)
        result = train_model(X, y, tmp_path)
        assert (tmp_path / "model.joblib").exists()
        assert (tmp_path / "classes.json").exists()
        assert result["classes"] == ["A", "B", "C"]
        assert result["accuracy"] > 0.8

    def test_refuses_a_single_class(self, tmp_path):
        X, y = sample_set(["A"], 10)
        with pytest.raises(ValueError, match="two different"):
            train_model(X, y, tmp_path)

    def test_refuses_a_class_with_one_sample(self, tmp_path):
        X, y = sample_set(["A", "B"], 10)
        X = np.vstack([X, X[:1]])
        y = y + ["C"]
        with pytest.raises(ValueError, match="too few"):
            train_model(X, y, tmp_path)

    def test_tiny_set_trains_without_a_holdout(self, tmp_path):
        """Too few samples to split is a warning case, not a failure."""
        X, y = sample_set(["A", "B"], 3)
        result = train_model(X, y, tmp_path)
        assert result["accuracy"] is None
        assert (tmp_path / "model.joblib").exists()

    def test_flags_under_recorded_letters(self, tmp_path):
        X, y = sample_set(["A", "B"], 6)
        assert set(train_model(X, y, tmp_path)["thin"]) == {"A", "B"}

    def test_saved_model_predicts_the_labels_it_was_given(self, tmp_path):
        import joblib
        X, y = sample_set(["A", "B", "space"], 20)
        train_model(X, y, tmp_path)
        model = joblib.load(tmp_path / "model.joblib")
        assert set(model.classes_) == {"A", "B", "space"}
        assert model.predict(X[:1])[0] == y[0]
