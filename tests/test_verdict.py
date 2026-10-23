"""Tests for src.verdict module."""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verdict import build_verdict


def _make_sample_results(c_acc: float, q_acc: float, with_entanglement: bool = True) -> list[dict]:
    """Helper to generate sample benchmark results dicts."""
    results = [
        {
            "name": "SVM",
            "type": "classical",
            "accuracy": c_acc,
            "f1": c_acc - 0.02,
            "train_time_s": 0.002,
            "n_params": 364,
            "predictions": [0, 1],
        },
        {
            "name": "Random Forest",
            "type": "classical",
            "accuracy": c_acc - 0.01,
            "f1": c_acc - 0.03,
            "train_time_s": 0.08,
            "n_params": 4500,
            "predictions": [0, 1],
        },
        {
            "name": "Quantum Kernel SVM",
            "type": "quantum",
            "accuracy": q_acc,
            "f1": q_acc - 0.02,
            "train_time_s": 0.35,
            "n_params": 4,
            "predictions": [0, 1],
            "entanglement": True,
        },
        {
            "name": "VQC",
            "type": "quantum",
            "accuracy": q_acc - 0.05,
            "f1": q_acc - 0.07,
            "train_time_s": 42.0,
            "n_params": 12,
            "predictions": [0, 1],
            "entanglement": True,
        },
    ]

    if with_entanglement:
        results.append({
            "name": "Quantum Kernel SVM",
            "type": "quantum",
            "accuracy": q_acc + 0.04,
            "f1": q_acc + 0.02,
            "train_time_s": 0.25,
            "n_params": 4,
            "predictions": [0, 1],
            "entanglement": False,
        })

    return results


def test_tie_case():
    """Verify statistical tie is identified when accuracy gap is below tie_margin."""
    # c_acc=0.95, q_acc=0.96 -> gap=0.01 (< 0.02)
    results = _make_sample_results(c_acc=0.95, q_acc=0.96, with_entanglement=False)
    verdict = build_verdict(results, tie_margin=0.02)

    assert "statistical tie" in verdict.lower()
    assert "0 trainable parameters" in verdict
    assert len(verdict.split(".")) >= 4


def test_quantum_wins_case():
    """Verify verdict acknowledges when quantum model significantly outperforms classical."""
    # c_acc=0.85, q_acc=0.95 -> gap=0.10 (> 0.02)
    results = _make_sample_results(c_acc=0.85, q_acc=0.95, with_entanglement=False)
    verdict = build_verdict(results, tie_margin=0.02)

    assert "quantum approach achieved the highest performance" in verdict.lower()
    assert "outperforming" in verdict.lower()
    assert "0 trainable parameters" in verdict


def test_classical_wins_case():
    """Verify verdict acknowledges when classical baselines outperform quantum."""
    # c_acc=0.97, q_acc=0.75 -> gap=-0.22
    results = _make_sample_results(c_acc=0.97, q_acc=0.75, with_entanglement=False)
    verdict = build_verdict(results, tie_margin=0.02)

    assert "classical baselines outperformed" in verdict.lower()
    assert "0 trainable parameters" in verdict


def test_missing_entanglement_case():
    """Verify build_verdict executes cleanly when entanglement on/off variants are omitted."""
    results = _make_sample_results(c_acc=0.96, q_acc=0.85, with_entanglement=False)
    # Strip entanglement keys from all models
    for r in results:
        r.pop("entanglement", None)

    verdict = build_verdict(results, tie_margin=0.02)
    assert isinstance(verdict, str)
    assert len(verdict) > 100
    assert "classical baselines outperformed" in verdict.lower()
    assert "0 trainable parameters" in verdict


def test_empty_results():
    """Verify build_verdict handles empty or incomplete result lists gracefully."""
    assert "No benchmark results available" in build_verdict([])
    assert "Insufficient model diversity" in build_verdict([{"name": "SVM", "type": "classical", "accuracy": 0.9}])
