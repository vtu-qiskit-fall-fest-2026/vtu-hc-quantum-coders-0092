"""Robustness and entanglement study across multiple random seeds and iterations."""

import time
from typing import Any, Dict, List, Sequence
import numpy as np

from src.data import load_breast_cancer_csv, preprocess_and_split
from src.quantum import train_quantum_kernel_svm, train_vqc


def run_entanglement_study(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_train: int = 120,
    maxiter_vqc: int = 300,
) -> Dict[str, Any]:
    """Run an entanglement study for Quantum Kernel SVM over multiple seeds and compare VQC iterations.

    Args:
        seeds: Random seeds for train/test split and subsampling.
        n_train: Number of training samples per run.
        maxiter_vqc: Higher iteration count for entangled VQC comparison.

    Returns:
        Dictionary containing mean and std for Quantum Kernel SVM (entangled vs non-entangled)
        and comparative metrics for VQC at maxiter=100 vs maxiter=300.
    """
    X, y = load_breast_cancer_csv()

    # 1. Quantum Kernel SVM across seeds
    acc_ent_list: List[float] = []
    acc_no_ent_list: List[float] = []

    print(f"Running Quantum Kernel SVM study over seeds {list(seeds)}...")
    for seed in seeds:
        data = preprocess_and_split(X, y, n_features=4, random_state=seed)

        res_ent = train_quantum_kernel_svm(data, entanglement=True, n_train=n_train, seed=seed)
        res_no_ent = train_quantum_kernel_svm(data, entanglement=False, n_train=n_train, seed=seed)

        acc_ent_list.append(res_ent["accuracy"])
        acc_no_ent_list.append(res_no_ent["accuracy"])

    mean_ent = float(np.mean(acc_ent_list))
    std_ent = float(np.std(acc_ent_list))
    mean_no_ent = float(np.mean(acc_no_ent_list))
    std_no_ent = float(np.std(acc_no_ent_list))

    # 2. Entangled VQC comparison (maxiter=100 vs maxiter=300)
    print("\nRunning Entangled VQC comparison (maxiter=100 vs maxiter=300)...")
    data_fixed = preprocess_and_split(X, y, n_features=4, random_state=42)

    t0 = time.perf_counter()
    vqc_100 = train_vqc(data_fixed, entanglement=True, maxiter=100, n_train=n_train, seed=42)
    t_100 = time.perf_counter() - t0
    print(f"  VQC maxiter=100 completed in {t_100:.1f}s (Acc: {vqc_100['accuracy']:.4f})")

    t0 = time.perf_counter()
    vqc_300 = train_vqc(data_fixed, entanglement=True, maxiter=maxiter_vqc, n_train=n_train, seed=42)
    t_300 = time.perf_counter() - t0
    print(f"  VQC maxiter={maxiter_vqc} completed in {t_300:.1f}s (Acc: {vqc_300['accuracy']:.4f})")

    return {
        "kernel_svm": {
            "entangled": {
                "mean": mean_ent,
                "std": std_ent,
                "accuracies": acc_ent_list,
            },
            "non_entangled": {
                "mean": mean_no_ent,
                "std": std_no_ent,
                "accuracies": acc_no_ent_list,
            },
            "seeds": list(seeds),
        },
        "vqc_entangled": {
            "maxiter_100": {
                "accuracy": vqc_100["accuracy"],
                "f1": vqc_100["f1"],
                "train_time_s": vqc_100["train_time_s"],
            },
            f"maxiter_{maxiter_vqc}": {
                "accuracy": vqc_300["accuracy"],
                "f1": vqc_300["f1"],
                "train_time_s": vqc_300["train_time_s"],
            },
        },
    }


if __name__ == "__main__":
    t0_study = time.perf_counter()
    results = run_entanglement_study(seeds=(0, 1, 2, 3, 4), n_train=120, maxiter_vqc=300)
    elapsed = time.perf_counter() - t0_study

    # Table 1: Quantum Kernel SVM Entanglement Study
    print("\n" + "=" * 65)
    print("QUANTUM KERNEL SVM: ENTANGLEMENT ROBUSTNESS STUDY (5 SEEDS)")
    print("=" * 65)
    k_res = results["kernel_svm"]
    print(f"{'Variant':<20} {'Mean Accuracy':<16} {'Std Dev':<14} {'Per-Seed Accuracies'}")
    print("-" * 65)
    ent_accs = ", ".join(f"{a:.4f}" for a in k_res["entangled"]["accuracies"])
    no_ent_accs = ", ".join(f"{a:.4f}" for a in k_res["non_entangled"]["accuracies"])
    print(f"{'Entangled (ZZ)':<20} {k_res['entangled']['mean']:<16.4f} {k_res['entangled']['std']:<14.4f} [{ent_accs}]")
    print(f"{'Non-Entangled (Z)':<20} {k_res['non_entangled']['mean']:<16.4f} {k_res['non_entangled']['std']:<14.4f} [{no_ent_accs}]")

    # Table 2: VQC Iteration Scaling Comparison
    print("\n" + "=" * 65)
    print("VQC (ENTANGLED): ITERATION SCALING COMPARISON")
    print("=" * 65)
    v_res = results["vqc_entangled"]
    print(f"{'Configuration':<20} {'Accuracy':<14} {'F1 Score':<14} {'Train Time (s)'}")
    print("-" * 65)
    print(f"{'COBYLA maxiter=100':<20} {v_res['maxiter_100']['accuracy']:<14.4f} {v_res['maxiter_100']['f1']:<14.4f} {v_res['maxiter_100']['train_time_s']:.2f}s")
    print(f"{'COBYLA maxiter=300':<20} {v_res['maxiter_300']['accuracy']:<14.4f} {v_res['maxiter_300']['f1']:<14.4f} {v_res['maxiter_300']['train_time_s']:.2f}s")
    print(f"\nTotal study time: {elapsed:.1f}s")
