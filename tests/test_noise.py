"""Tests for src.noise module."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data import load_breast_cancer_csv, preprocess_and_split
from src.noise import run_noise_sweep
from src.quantum import train_quantum_kernel_svm, train_vqc

EXPECTED_NOISE_KEYS = {"model_name", "entanglement", "noise_level", "accuracy"}


def test_noise_sweep_keys_and_accuracy():
    """Verify noise sweep output keys, fake backend inclusion, and noiseless match at p=0.0."""
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    # Fast model train with small samples
    res_svm = train_quantum_kernel_svm(data, entanglement=False, n_train=20, seed=42)

    # Quick noise sweep with tiny n_test and shots
    sweep = run_noise_sweep(
        data,
        res_svm,
        noise_levels=(0.0, 0.01),
        shots=200,
        seed=42,
        n_test=15,
    )

    assert isinstance(sweep, list)
    assert len(sweep) == 3  # 0.0, 0.01, and fake_backend

    for item in sweep:
        assert EXPECTED_NOISE_KEYS.issubset(set(item.keys()))
        assert item["model_name"] == "Quantum Kernel SVM"
        assert item["entanglement"] is False
        assert 0.0 <= item["accuracy"] <= 1.0

    # Verify fake_backend is present
    noise_levels = [item["noise_level"] for item in sweep]
    assert "fake_backend" in noise_levels
    assert 0.0 in noise_levels

    # Check that at noise level 0.0, accuracy matches noiseless accuracy within small tolerance
    idx_test_sub, _ = train_test_split(
        np.arange(len(data["y_test"])),
        train_size=15,
        random_state=42,
        stratify=data["y_test"],
    )
    noiseless_preds_sub = np.array(res_svm["predictions"])[idx_test_sub]
    noiseless_acc = float(accuracy_score(data["y_test"][idx_test_sub], noiseless_preds_sub))

    p0_acc = sweep[0]["accuracy"]
    assert np.isclose(p0_acc, noiseless_acc, atol=0.05), (
        f"p=0.0 accuracy {p0_acc} differs from noiseless {noiseless_acc}"
    )


def test_vqc_noise_sweep_quick():
    """Verify VQC re-evaluation under noise sweep."""
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    res_vqc = train_vqc(data, entanglement=False, reps=1, maxiter=10, n_train=20, seed=42)
    sweep = run_noise_sweep(
        data,
        res_vqc,
        noise_levels=(0.0, 0.01),
        shots=100,
        seed=42,
        n_test=10,
    )

    assert len(sweep) == 3
    for item in sweep:
        assert EXPECTED_NOISE_KEYS.issubset(set(item.keys()))
        assert 0.0 <= item["accuracy"] <= 1.0
