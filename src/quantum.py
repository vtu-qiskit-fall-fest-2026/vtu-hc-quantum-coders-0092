"""Quantum machine learning models (VQC and Quantum Kernel SVM)."""

from typing import Any, Dict, Optional
import numpy as np


def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    entanglement: bool = True,
    maxiter: int = 50,
    sampler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train a Variational Quantum Classifier (VQC) with configurable entanglement.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        entanglement: Whether to enable entanglement in feature map and ansatz.
        maxiter: Maximum optimization iterations.
        sampler: Primitive sampler (AerSampler or IBM backend sampler).

    Returns:
        Dictionary containing trained model, test score, loss history, and metadata.
    """
    # TODO: Build feature map and ansatz with or without entanglement
    # TODO: Fit VQC using AerSampler or hardware sampler with COBYLA / SPSA optimizer
    raise NotImplementedError("TODO: Implement train_vqc")


def train_quantum_kernel_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    entanglement: bool = True,
    sampler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train a Support Vector Classifier with a Quantum Kernel.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        entanglement: Whether to include entanglement in the quantum feature map.
        sampler: Primitive sampler or fidelity estimator.

    Returns:
        Dictionary containing trained QSVC model, test score, and kernel matrix metrics.
    """
    # TODO: Construct fidelity quantum kernel with configurable entanglement
    # TODO: Compute kernel matrices and fit scikit-learn SVC(kernel='precomputed') or QSVC
    raise NotImplementedError("TODO: Implement train_quantum_kernel_svm")
