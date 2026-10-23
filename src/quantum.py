"""Quantum machine learning models (VQC and Quantum Kernel SVM) using Qiskit."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.optimizers import COBYLA


def _get_feature_map(n_qubits: int, entanglement: bool, reps: int = 2) -> QuantumCircuit:
    """Construct quantum feature map with or without entanglement.

    When entanglement=True: zz_feature_map with linear 2-qubit CX gates.
    When entanglement=False: z_feature_map with single-qubit rotations only (0 CX gates).
    """
    if entanglement:
        return zz_feature_map(n_qubits, reps=reps, entanglement="linear")
    return z_feature_map(n_qubits, reps=reps)


def _get_ansatz(n_qubits: int, entanglement: bool, reps: int = 2) -> QuantumCircuit:
    """Construct RealAmplitudes ansatz with or without entanglement.

    When entanglement=True: linear 2-qubit CX entanglement layers.
    When entanglement=False: entanglement=[] (empty list, 0 CX gates, only single-qubit RY gates).
    """
    if entanglement:
        return real_amplitudes(n_qubits, reps=reps, entanglement="linear")
    return real_amplitudes(n_qubits, reps=reps, entanglement=[])


def _subsample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_train: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stratified subsampling of training data to keep quantum simulation fast."""
    n_total = len(y_train)
    if n_train is not None and 0 < n_train < n_total:
        idx_sub, _ = train_test_split(
            np.arange(n_total),
            train_size=n_train,
            random_state=seed,
            stratify=y_train,
        )
        return X_train[idx_sub], y_train[idx_sub], int(n_train)
    return X_train, y_train, int(n_total)


def train_vqc(
    data: Dict[str, Any],
    entanglement: bool = True,
    reps: int = 2,
    maxiter: int = 100,
    n_train: int = 120,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a Variational Quantum Classifier (VQC) with configurable entanglement.

    Args:
        data: Preprocessed data dict from src.data.preprocess_and_split.
        entanglement: Whether to enable two-qubit entangling gates.
        reps: Circuit repetitions for feature map and ansatz.
        maxiter: Maximum iterations for COBYLA optimizer.
        n_train: Number of stratified training samples to use.
        seed: Random seed for subsampling and optimizer.

    Returns:
        Dictionary containing benchmark metrics, circuit metadata, and predictions.
    """
    n_qubits = int(data.get("n_features", data["X_train"].shape[1]))
    X_train_sub, y_train_sub, n_train_used = _subsample_train(
        data["X_train"], data["y_train"], n_train=n_train, seed=seed
    )

    feature_map = _get_feature_map(n_qubits, entanglement=entanglement, reps=reps)
    ansatz = _get_ansatz(n_qubits, entanglement=entanglement, reps=reps)

    # Ansatz parameters count (trainable variational weights)
    n_params = int(ansatz.num_parameters)

    # Transpile full circuit on AerSimulator to count depth and 2-qubit gates
    backend = AerSimulator()
    full_circuit = feature_map.compose(ansatz)
    t_circuit = transpile(full_circuit, backend=backend, optimization_level=1)
    circuit_depth = int(t_circuit.depth())
    two_qubit_gate_count = int(sum(1 for inst in t_circuit.data if inst.operation.num_qubits == 2))

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    optimizer = COBYLA(maxiter=maxiter)
    sampler = AerSampler()

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        pass_manager=pm,
    )

    t0 = time.perf_counter()
    vqc.fit(X_train_sub, y_train_sub)
    train_time_s = time.perf_counter() - t0

    if train_time_s > 300.0:
        warnings.warn(
            f"VQC training took {train_time_s:.1f}s (> 5 minutes). "
            "Consider lowering maxiter or n_train for faster execution.",
            UserWarning,
            stacklevel=2,
        )

    preds = vqc.predict(data["X_test"])
    preds_int = [int(p) for p in preds]

    accuracy = float(accuracy_score(data["y_test"], preds_int))
    f1 = float(f1_score(data["y_test"], preds_int, pos_label=1))

    return {
        "name": "VQC",
        "type": "quantum",
        "accuracy": accuracy,
        "f1": f1,
        "train_time_s": float(train_time_s),
        "n_params": n_params,
        "predictions": preds_int,
        "entanglement": bool(entanglement),
        "n_qubits": n_qubits,
        "n_train_used": n_train_used,
        "circuit_depth": circuit_depth,
        "two_qubit_gate_count": two_qubit_gate_count,
    }


def train_quantum_kernel_svm(
    data: Dict[str, Any],
    entanglement: bool = True,
    n_train: int = 120,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a Support Vector Classifier with a Quantum Kernel.

    Args:
        data: Preprocessed data dict from src.data.preprocess_and_split.
        entanglement: Whether to enable two-qubit entangling gates in the feature map.
        n_train: Number of stratified training samples to use.
        seed: Random seed for subsampling.

    Returns:
        Dictionary containing benchmark metrics, circuit metadata, and predictions.
    """
    n_qubits = int(data.get("n_features", data["X_train"].shape[1]))
    X_train_sub, y_train_sub, n_train_used = _subsample_train(
        data["X_train"], data["y_train"], n_train=n_train, seed=seed
    )

    feature_map = _get_feature_map(n_qubits, entanglement=entanglement, reps=2)

    # For Quantum Kernel SVM, n_params represents the number of encoding parameters
    # in the feature map circuit (equal to input feature dimensions mapped to rotation angles).
    n_params = int(feature_map.num_parameters)

    # Transpile feature map to count circuit depth and 2-qubit gates
    backend = AerSimulator()
    t_fm = transpile(feature_map, backend=backend, optimization_level=1)
    circuit_depth = int(t_fm.depth())
    two_qubit_gate_count = int(sum(1 for inst in t_fm.data if inst.operation.num_qubits == 2))

    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel)

    t0 = time.perf_counter()
    qsvc.fit(X_train_sub, y_train_sub)
    train_time_s = time.perf_counter() - t0

    if train_time_s > 300.0:
        warnings.warn(
            f"Quantum Kernel SVM training took {train_time_s:.1f}s (> 5 minutes). "
            "Consider lowering n_train for faster execution.",
            UserWarning,
            stacklevel=2,
        )

    preds = qsvc.predict(data["X_test"])
    preds_int = [int(p) for p in preds]

    accuracy = float(accuracy_score(data["y_test"], preds_int))
    f1 = float(f1_score(data["y_test"], preds_int, pos_label=1))

    return {
        "name": "Quantum Kernel SVM",
        "type": "quantum",
        "accuracy": accuracy,
        "f1": f1,
        "train_time_s": float(train_time_s),
        "n_params": n_params,
        "predictions": preds_int,
        "entanglement": bool(entanglement),
        "n_qubits": n_qubits,
        "n_train_used": n_train_used,
        "circuit_depth": circuit_depth,
        "two_qubit_gate_count": two_qubit_gate_count,
    }


def save_circuit_diagrams(out_dir: str | Path = "docs/") -> None:
    """Save PNG diagrams of the feature map and ansatz (entangled and non-entangled).

    Args:
        out_dir: Output directory where PNG files are saved.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_qubits = 4
    fm_ent = _get_feature_map(n_qubits, entanglement=True, reps=2)
    fm_no = _get_feature_map(n_qubits, entanglement=False, reps=2)
    ansatz_ent = _get_ansatz(n_qubits, entanglement=True, reps=2)
    ansatz_no = _get_ansatz(n_qubits, entanglement=False, reps=2)

    fm_ent.draw("mpl", filename=str(out_path / "feature_map_entangled.png"))
    fm_no.draw("mpl", filename=str(out_path / "feature_map_no_entanglement.png"))
    ansatz_ent.draw("mpl", filename=str(out_path / "ansatz_entangled.png"))
    ansatz_no.draw("mpl", filename=str(out_path / "ansatz_no_entanglement.png"))
    print(f"Saved circuit diagrams to '{out_path}/'")


if __name__ == "__main__":
    from src.data import load_breast_cancer_csv, preprocess_and_split

    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    print("Running quantum model benchmark (4 combinations)...")
    results = [
        train_vqc(data, entanglement=True, maxiter=100, n_train=120),
        train_vqc(data, entanglement=False, maxiter=100, n_train=120),
        train_quantum_kernel_svm(data, entanglement=True, n_train=120),
        train_quantum_kernel_svm(data, entanglement=False, n_train=120),
    ]

    header = (
        f"{'Name':<20} {'Entanglement':<14} {'Accuracy':<10} {'F1':<10} "
        f"{'Train Time (s)':<16} {'Params':<8} {'2Q Gates':<10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res['name']:<20} {str(res['entanglement']):<14} {res['accuracy']:<10.4f} "
            f"{res['f1']:<10.4f} {res['train_time_s']:<16.4f} {res['n_params']:<8} "
            f"{res['two_qubit_gate_count']:<10}"
        )

    print("\nGenerating circuit diagrams...")
    save_circuit_diagrams("docs/")
