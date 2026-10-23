"""Tests for src.data module."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data import load_breast_cancer_csv, preprocess_and_split


def test_load_breast_cancer_csv():
    """Check: 569 rows and 30 features after loading, and y contains only 0 and 1."""
    X, y = load_breast_cancer_csv()

    assert X.shape == (569, 30), f"Expected (569, 30), got {X.shape}"
    assert len(y) == 569, f"Expected 569 labels, got {len(y)}"
    assert set(np.unique(y)) == {0, 1}, f"Expected labels {{0, 1}}, got {set(np.unique(y))}"
    assert "id" not in X.columns
    assert not any("unnamed" in c.lower() for c in X.columns)


def test_preprocess_and_split_shapes_and_features():
    """Check: train/test split shapes are right, and output has exactly n_features columns."""
    X, y = load_breast_cancer_csv()

    # Default n_features=4, test_size=0.25 -> 426 train (75%), 143 test (25%)
    data = preprocess_and_split(X, y, n_features=4, test_size=0.25, random_state=42)

    assert data["X_train"].shape == (426, 4)
    assert data["X_test"].shape == (143, 4)
    assert data["y_train"].shape == (426,)
    assert data["y_test"].shape == (143,)
    assert data["n_features"] == 4
    assert 0.0 < data["explained_variance"] <= 1.0

    # Custom n_features
    for n_feat in [2, 6]:
        data_custom = preprocess_and_split(X, y, n_features=n_feat, test_size=0.25, random_state=42)
        assert data_custom["X_train"].shape[1] == n_feat
        assert data_custom["X_test"].shape[1] == n_feat
        assert data_custom["n_features"] == n_feat


def test_train_values_range_within_0_to_pi():
    """Check: train set values are strictly within [0, pi] (test values may slightly exceed it)."""
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4, test_size=0.25, random_state=42)

    X_train = data["X_train"]
    assert X_train.min() >= -1e-9, f"Train min {X_train.min()} is below 0"
    assert X_train.max() <= np.pi + 1e-9, f"Train max {X_train.max()} exceeds pi"
    assert np.isclose(X_train.min(), 0.0, atol=1e-5)
    assert np.isclose(X_train.max(), np.pi, atol=1e-5)


def test_split_reproducibility():
    """Check: calling the function twice with the same random_state gives identical splits."""
    X, y = load_breast_cancer_csv()

    data_1 = preprocess_and_split(X, y, n_features=4, test_size=0.25, random_state=42)
    data_2 = preprocess_and_split(X, y, n_features=4, test_size=0.25, random_state=42)

    np.testing.assert_array_equal(data_1["X_train"], data_2["X_train"])
    np.testing.assert_array_equal(data_1["X_test"], data_2["X_test"])
    np.testing.assert_array_equal(data_1["y_train"], data_2["y_train"])
    np.testing.assert_array_equal(data_1["y_test"], data_2["y_test"])
    assert data_1["explained_variance"] == data_2["explained_variance"]
