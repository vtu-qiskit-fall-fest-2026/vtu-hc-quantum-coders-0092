"""Tests for src.quantum module."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_breast_cancer_csv, preprocess_and_split
from src.quantum import save_circuit_diagrams, train_quantum_kernel_svm, train_vqc

EXPECTED_KEYS = {
    "name",
    "type",
    "accuracy",
    "f1",
    "train_time_s",
    "n_params",
    "predictions",
    "entanglement",
    "n_qubits",
    "n_train_used",
    "circuit_depth",
    "two_qubit_gate_count",
}


def test_vqc_entanglement_and_keys():
    """Verify VQC returns expected keys, valid metric range, and exact 2-qubit gate counts."""
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    # Fast test run with tiny maxiter and n_train=30
    res_ent = train_vqc(data, entanglement=True, reps=1, maxiter=10, n_train=30, seed=42)
    res_no_ent = train_vqc(data, entanglement=False, reps=1, maxiter=10, n_train=30, seed=42)

    for res in [res_ent, res_no_ent]:
        assert set(res.keys()) == EXPECTED_KEYS
        assert res["type"] == "quantum"
        assert res["name"] == "VQC"
        assert 0.0 <= res["accuracy"] <= 1.0
        assert 0.0 <= res["f1"] <= 1.0
        assert res["n_qubits"] == 4
        assert res["n_train_used"] == 30
        assert res["train_time_s"] >= 0.0
        assert res["n_params"] > 0
        assert len(res["predictions"]) == len(data["y_test"])

    # Entanglement test: zero 2Q gates when False, strictly positive when True
    assert res_no_ent["two_qubit_gate_count"] == 0, "entanglement=False must have 0 two-qubit gates"
    assert res_ent["two_qubit_gate_count"] > 0, "entanglement=True must have > 0 two-qubit gates"
    assert res_no_ent["entanglement"] is False
    assert res_ent["entanglement"] is True


def test_quantum_kernel_svm_entanglement_and_keys():
    """Verify Quantum Kernel SVM returns expected keys, valid metrics, and 2-qubit gate counts."""
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    # Fast test run with n_train=30
    res_ent = train_quantum_kernel_svm(data, entanglement=True, n_train=30, seed=42)
    res_no_ent = train_quantum_kernel_svm(data, entanglement=False, n_train=30, seed=42)

    for res in [res_ent, res_no_ent]:
        assert set(res.keys()) == EXPECTED_KEYS
        assert res["type"] == "quantum"
        assert res["name"] == "Quantum Kernel SVM"
        assert 0.0 <= res["accuracy"] <= 1.0
        assert 0.0 <= res["f1"] <= 1.0
        assert res["n_qubits"] == 4
        assert res["n_train_used"] == 30
        assert res["train_time_s"] >= 0.0
        assert res["n_params"] == 4  # 4 feature dimensions
        assert len(res["predictions"]) == len(data["y_test"])

    assert res_no_ent["two_qubit_gate_count"] == 0, "entanglement=False must have 0 two-qubit gates"
    assert res_ent["two_qubit_gate_count"] > 0, "entanglement=True must have > 0 two-qubit gates"
    assert res_no_ent["entanglement"] is False
    assert res_ent["entanglement"] is True


def test_save_circuit_diagrams(tmp_path):
    """Verify save_circuit_diagrams exports all 4 PNG files."""
    save_circuit_diagrams(tmp_path)
    assert (tmp_path / "feature_map_entangled.png").exists()
    assert (tmp_path / "feature_map_no_entanglement.png").exists()
    assert (tmp_path / "ansatz_entangled.png").exists()
    assert (tmp_path / "ansatz_no_entanglement.png").exists()
