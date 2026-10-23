"""Smoke tests verifying module imports and a tiny end-to-end VQC training run."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path when running pytest directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

import src
import src.data
import src.classical
import src.quantum
import src.noise
import src.verdict
import src.ibm_hardware
import app


def test_imports():
    """Verify that all project modules can be imported without error."""
    assert src is not None
    assert src.data is not None
    assert src.classical is not None
    assert src.quantum is not None
    assert src.noise is not None
    assert src.verdict is not None
    assert src.ibm_hardware is not None
    assert app is not None


def test_tiny_vqc_smoke():
    """Verify that a 2-qubit VQC builds, fits on toy data using COBYLA and Aer, and predicts."""
    num_qubits = 2
    feature_map = zz_feature_map(feature_dimension=num_qubits, reps=1)
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=1)
    optimizer = COBYLA(maxiter=10)
    sampler = AerSampler()

    backend = AerSimulator()
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        pass_manager=pm,
    )

    np.random.seed(42)
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=10)

    vqc.fit(X, y)
    preds = vqc.predict(X[:2])

    assert preds is not None
    assert len(preds) == 2
