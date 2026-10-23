"""Tests for src.classical module."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split
from src.classical import run_classical_on_full_features, train_classical_models
from src.data import load_breast_cancer_csv, preprocess_and_split


REQUIRED_KEYS = {"name", "type", "accuracy", "f1", "train_time_s", "n_params", "predictions"}


def test_train_classical_models_structure_and_keys():
    """Verify three results returned and every dict has the exact required keys."""
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    results = train_classical_models(data, random_state=42)

    assert len(results) == 3, f"Expected 3 models, got {len(results)}"
    model_names = [r["name"] for r in results]
    assert set(model_names) == {"SVM", "Random Forest", "MLP"}

    for res in results:
        assert set(res.keys()) == REQUIRED_KEYS, f"Keys mismatch in {res['name']}: {set(res.keys())}"
        assert res["type"] == "classical"
        assert isinstance(res["name"], str)
        assert isinstance(res["accuracy"], float)
        assert isinstance(res["f1"], float)
        assert isinstance(res["train_time_s"], float)
        assert isinstance(res["n_params"], int)
        assert isinstance(res["predictions"], list)
        assert len(res["predictions"]) == len(data["y_test"])
        assert all(isinstance(p, int) for p in res["predictions"])


def test_accuracy_and_f1_metrics():
    """Verify accuracy and f1 are between 0 and 1, and accuracy is above 0.85."""
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    results = train_classical_models(data, random_state=42)

    for res in results:
        assert 0.0 <= res["accuracy"] <= 1.0, f"{res['name']} accuracy out of [0, 1]"
        assert 0.0 <= res["f1"] <= 1.0, f"{res['name']} f1 out of [0, 1]"
        assert res["accuracy"] > 0.85, f"{res['name']} accuracy {res['accuracy']} is not above 0.85"
        assert res["n_params"] > 0, f"{res['name']} n_params should be > 0"
        assert res["train_time_s"] >= 0.0, f"{res['name']} train_time_s should be non-negative"


def test_seed_reproducibility():
    """Verify two runs with the same random seed produce identical accuracy and predictions."""
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    results_run1 = train_classical_models(data, random_state=42)
    results_run2 = train_classical_models(data, random_state=42)

    for r1, r2 in zip(results_run1, results_run2):
        assert r1["name"] == r2["name"]
        assert r1["accuracy"] == r2["accuracy"], f"Mismatch in {r1['name']} accuracy across runs"
        assert r1["f1"] == r2["f1"], f"Mismatch in {r1['name']} f1 across runs"
        assert r1["predictions"] == r2["predictions"], f"Mismatch in {r1['name']} predictions"
        assert r1["n_params"] == r2["n_params"]


def test_run_classical_on_full_features():
    """Verify full feature reference runs cleanly and returns type='classical_full'."""
    X, y = load_breast_cancer_csv()
    X_tr_full, X_te_full, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    full_results = run_classical_on_full_features(X_tr_full, X_te_full, y_tr, y_te, random_state=42)

    assert len(full_results) == 3
    for res in full_results:
        assert set(res.keys()) == REQUIRED_KEYS
        assert res["type"] == "classical_full"
        assert res["accuracy"] > 0.85
        assert 0.0 <= res["f1"] <= 1.0
