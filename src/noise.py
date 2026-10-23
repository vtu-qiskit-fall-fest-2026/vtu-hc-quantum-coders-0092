"""Noise sweep testing for quantum models using depolarizing noise and fake IBM hardware."""

import time
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime.fake_provider import FakeTorino
from src.quantum import predict_with_sampler


def build_depolarizing_noise_model(p: float) -> NoiseModel:
    """Build a NoiseModel with depolarizing error p on 1-qubit gates and 10*p on 2-qubit gates.

    Args:
        p: Single-qubit gate depolarizing error probability.

    Returns:
        NoiseModel instance.
    """
    noise_model = NoiseModel()
    if p > 0.0:
        error_1q = depolarizing_error(p, 1)
        error_2q = depolarizing_error(min(10.0 * p, 0.75), 2)
        one_q_gates = ["u", "p", "r", "rx", "ry", "rz", "h", "x", "sx"]
        two_q_gates = ["cx", "cz", "swap", "ecr"]
        noise_model.add_all_qubit_quantum_error(error_1q, one_q_gates)
        noise_model.add_all_qubit_quantum_error(error_2q, two_q_gates)
    return noise_model


def get_fake_backend_noise_model() -> Tuple[str, NoiseModel]:
    """Retrieve NoiseModel from the Heron-family FakeTorino backend."""
    backend = FakeTorino()
    return backend.name, NoiseModel.from_backend(backend)


def run_noise_sweep(
    data: Dict[str, Any],
    model_result: Dict[str, Any],
    noise_levels: Sequence[float] = (0.0, 0.002, 0.005, 0.01, 0.02),
    shots: int = 1024,
    seed: int = 42,
    n_test: int = 60,
) -> List[Dict[str, Any]]:
    """Evaluate an already trained quantum model under simulated depolarizing noise and fake backend noise.

    Args:
        data: Preprocessed data dict from src.data.preprocess_and_split.
        model_result: Result dict from train_vqc or train_quantum_kernel_svm.
        noise_levels: Sequence of single-qubit depolarizing error rates.
        shots: Measurement shots per circuit evaluation.
        seed: Random seed for subsampling and noisy sampling.
        n_test: Fixed number of stratified test samples for fast evaluation.

    Returns:
        List of dicts: model_name, entanglement, noise_level, accuracy.
    """
    n_total_test = len(data["y_test"])
    if n_test is not None and 0 < n_test < n_total_test:
        idx_test, _ = train_test_split(
            np.arange(n_total_test),
            train_size=n_test,
            random_state=seed,
            stratify=data["y_test"],
        )
        X_test_sub = data["X_test"][idx_test]
        y_test_sub = data["y_test"][idx_test]
    else:
        X_test_sub = data["X_test"]
        y_test_sub = data["y_test"]

    results: List[Dict[str, Any]] = []

    # 1. Depolarizing noise levels
    for p in noise_levels:
        if p == 0.0:
            sampler = AerSampler(default_shots=shots, seed=seed)
        else:
            nm = build_depolarizing_noise_model(float(p))
            sampler = AerSampler(
                default_shots=shots,
                seed=seed,
                options={"backend_options": {"noise_model": nm}},
            )

        preds = predict_with_sampler(model_result, X_test_sub, sampler)
        acc = float(accuracy_score(y_test_sub, preds))

        results.append({
            "model_name": model_result["name"],
            "name": model_result["name"],
            "entanglement": bool(model_result["entanglement"]),
            "noise_level": float(p),
            "accuracy": acc,
        })

    # 2. Fake IBM Heron backend noise model (FakeTorino)
    _, nm_fake = get_fake_backend_noise_model()
    sampler_fake = AerSampler(
        default_shots=shots,
        seed=seed,
        options={"backend_options": {"noise_model": nm_fake}},
    )
    preds_fake = predict_with_sampler(model_result, X_test_sub, sampler_fake)
    acc_fake = float(accuracy_score(y_test_sub, preds_fake))

    results.append({
        "model_name": model_result["name"],
        "name": model_result["name"],
        "entanglement": bool(model_result["entanglement"]),
        "noise_level": "fake_backend",
        "accuracy": acc_fake,
    })

    return results


def sanity_check_noise(
    data: Dict[str, Any],
    vqc_model: Dict[str, Any],
    svm_model: Dict[str, Any],
    noise_levels: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.4),
    n_samples: int = 5,
) -> None:
    """Sanity-check noise application: basis gate coverage, noise sweep, side-by-side probabilities, and sampler verification."""
    from qiskit import transpile
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    print("=" * 75)
    print("DEBUG & SANITY-CHECK: NOISE MODEL APPLICATION")
    print("=" * 75)

    # 1. Gate names with errors vs transpiled circuit gate names
    nm_test = build_depolarizing_noise_model(0.1)
    s_test = AerSampler(options={"backend_options": {"noise_model": nm_test}})
    basis_with_errors = sorted(nm_test.noise_instructions)

    fm_no = svm_model["_feature_map"]
    ans_no = vqc_model["_ansatz"]
    vqc_circ = fm_no.compose(ans_no)
    vqc_circ.measure_all()
    t_vqc = transpile(vqc_circ, backend=s_test._backend)

    fid = ComputeUncompute(sampler=s_test)
    kernel_circ = fid.create_fidelity_circuit(fm_no, fm_no)
    t_kernel = transpile(kernel_circ, backend=s_test._backend)

    vqc_ops = dict(t_vqc.count_ops())
    kernel_ops = dict(t_kernel.count_ops())

    print("\n1. BASIS GATES & CIRCUIT COVERAGE:")
    print(f"   NoiseModel gates with errors: {basis_with_errors}")
    print(f"   Transpiled Non-Entangled VQC ops: {vqc_ops}")
    print(f"   Transpiled Non-Entangled Kernel ops: {kernel_ops}")

    vqc_quantum_gates = [g for g in vqc_ops.keys() if g not in ("measure", "barrier")]
    kernel_quantum_gates = [g for g in kernel_ops.keys() if g not in ("measure", "barrier")]
    all_covered = all(g in basis_with_errors for g in set(vqc_quantum_gates + kernel_quantum_gates))

    print(f"   VQC quantum gates: {vqc_quantum_gates}")
    print(f"   Kernel quantum gates: {kernel_quantum_gates}")
    if all_covered:
        print("   -> YES, every gate in the circuits is covered by the noise model.")
    else:
        print("   -> NO, some gates are NOT covered by the noise model.")

    # 2. Accuracy at noise p = 0.0, 0.05, 0.1, 0.2, 0.4
    print("\n2. ACCURACY AT NOISE LEVELS (p = 0.0, 0.05, 0.1, 0.2, 0.4):")
    X_test_sub = data["X_test"][:30]
    y_test_sub = data["y_test"][:30]

    vqc_accs = {}
    svm_accs = {}
    for p in noise_levels:
        if p == 0.0:
            s_eval = AerSampler(default_shots=256, seed=42)
        else:
            nm = build_depolarizing_noise_model(p)
            s_eval = AerSampler(default_shots=256, seed=42, options={"backend_options": {"noise_model": nm}})

        preds_v = predict_with_sampler(vqc_model, X_test_sub, s_eval)
        acc_v = float(accuracy_score(y_test_sub, preds_v))
        vqc_accs[p] = acc_v

        preds_s = predict_with_sampler(svm_model, X_test_sub, s_eval)
        acc_s = float(accuracy_score(y_test_sub, preds_s))
        svm_accs[p] = acc_s

    print(f"   {'p':<8} {'VQC (No Ent) Accuracy':<25} {'Kernel SVM (No Ent) Accuracy':<30}")
    print("   " + "-" * 63)
    for p in noise_levels:
        print(f"   {p:<8.2f} {vqc_accs[p]:<25.4f} {svm_accs[p]:<30.4f}")

    # 3. 5 test samples side-by-side probabilities at p=0.0 and p=0.2
    print("\n3. VQC OUTPUT PROBABILITIES FOR 5 TEST SAMPLES (p=0.0 vs p=0.2 SIDE BY SIDE):")
    X_5 = data["X_test"][:n_samples]
    y_5 = data["y_test"][:n_samples]

    s_0 = AerSampler(default_shots=1024, seed=42)
    nm_2 = build_depolarizing_noise_model(0.2)
    s_2 = AerSampler(default_shots=1024, seed=42, options={"backend_options": {"noise_model": nm_2}})

    vqc_obj = vqc_model["_model"]
    vqc_obj.neural_network.sampler = s_0
    probs_0 = vqc_obj.neural_network.forward(X_5, vqc_obj.weights)

    vqc_obj.neural_network.sampler = s_2
    probs_2 = vqc_obj.neural_network.forward(X_5, vqc_obj.weights)

    print(f"   {'Sample':<8} {'Label':<8} {'p = 0.0 [P(0), P(1)]':<26} {'p = 0.2 [P(0), P(1)]':<26}")
    print("   " + "-" * 68)
    for i in range(n_samples):
        p0_str = f"[{probs_0[i][0]:.4f}, {probs_0[i][1]:.4f}]"
        p2_str = f"[{probs_2[i][0]:.4f}, {probs_2[i][1]:.4f}]"
        print(f"   {i:<8} {y_5[i]:<8} {p0_str:<26} {p2_str:<26}")

    # 4. Sampler noise model verification and transpilation
    print("\n4. SAMPLER NOISE MODEL VERIFICATION & TRANSPILATION:")
    nm_check = build_depolarizing_noise_model(0.05)
    s_check = AerSampler(options={"backend_options": {"noise_model": nm_check}})
    backend_nm = getattr(s_check._backend.options, "noise_model", None)
    nm_received = backend_nm is not None
    print(f"   - Does AerSampler receive the noise model: {nm_received}")
    print(f"   - Backend noise model instructions: {getattr(backend_nm, 'noise_instructions', None)}")
    print("   - Transpilation: AerSampler executes via AerSimulator and AerCompiler.")
    print("     Incoming circuit gates are checked against AerSimulator basis gates.")
    print("     Since all quantum gates (ry, h, p) are directly registered in the NoiseModel,")
    print("     each gate has the depolarizing error channel applied directly on execution.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    from src.data import load_breast_cancer_csv, preprocess_and_split
    from src.quantum import train_quantum_kernel_svm, train_vqc

    print("Loading data and training baseline noiseless quantum models...")
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    # Train noiseless models
    vqc_ent = train_vqc(data, entanglement=True, maxiter=100, n_train=120)
    vqc_no_ent = train_vqc(data, entanglement=False, maxiter=100, n_train=120)
    svm_ent = train_quantum_kernel_svm(data, entanglement=True, n_train=120)
    svm_no_ent = train_quantum_kernel_svm(data, entanglement=False, n_train=120)

    # Sanity check: Extreme noise sweep, basis gates, and probability comparison
    sanity_check_noise(data, vqc_no_ent, svm_no_ent, noise_levels=(0.0, 0.05, 0.1, 0.2, 0.4), n_samples=5)

    # Evaluate noise sweep with n_test=30 and shots=256 for fast execution (~2-3 mins total)
    print("Running noise sweep across all 4 quantum models (shots=256, n_test=30)...")
    t0_all = time.perf_counter()

    all_sweep_results = []
    models = [vqc_ent, vqc_no_ent, svm_ent, svm_no_ent]
    for m in models:
        t0 = time.perf_counter()
        sweep = run_noise_sweep(
            data,
            m,
            noise_levels=(0.0, 0.002, 0.005, 0.01, 0.02),
            shots=256,
            n_test=30,
        )
        elapsed = time.perf_counter() - t0
        print(f"Evaluated {m['name']} (entanglement={m['entanglement']}) in {elapsed:.1f}s")
        all_sweep_results.extend(sweep)

    total_time = time.perf_counter() - t0_all
    print(f"\nCompleted noise sweep in {total_time:.1f}s\n")

    header = f"{'Model':<20} {'Entanglement':<14} {'Noise Level':<15} {'Accuracy':<10}"
    print(header)
    print("-" * len(header))
    for r in all_sweep_results:
        lvl_str = f"{r['noise_level']:.4f}" if isinstance(r["noise_level"], float) else str(r["noise_level"])
        print(f"{r['model_name']:<20} {str(r['entanglement']):<14} {lvl_str:<15} {r['accuracy']:<10.4f}")
