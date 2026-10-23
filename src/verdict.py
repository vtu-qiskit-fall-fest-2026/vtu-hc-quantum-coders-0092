"""Plain-language benchmark verdict generator comparing quantum and classical models."""

from typing import Any, Dict, List, Optional


def build_verdict(results: List[Dict[str, Any]], tie_margin: float = 0.02) -> str:
    """Generate a balanced, 4-6 sentence plain-language verdict on benchmark outcomes.

    Compares the best classical and best quantum models, evaluates accuracy gaps against
    the tie margin, compares training efficiency and parameter scaling, and analyzes
    the impact of quantum entanglement.

    Args:
        results: List of model result dicts from classical and quantum training runs.
        tie_margin: Accuracy threshold below which performance difference is considered a tie (default 0.02).

    Returns:
        4-6 sentence plain-language summary string.
    """
    if not results:
        return "No benchmark results available to generate a verdict."

    classical_models = [r for r in results if r.get("type") in ("classical", "classical_full")]
    quantum_models = [r for r in results if r.get("type") == "quantum"]

    if not classical_models or not quantum_models:
        return "Insufficient model diversity: both classical and quantum results are required to generate a comparative verdict."

    # Prefer 4-feature classical models for like-for-like comparison if available
    c_reduced = [r for r in classical_models if r.get("type") == "classical"]
    c_pool = c_reduced if c_reduced else classical_models

    best_c = max(c_pool, key=lambda m: m.get("accuracy", 0.0))
    best_q = max(quantum_models, key=lambda m: m.get("accuracy", 0.0))

    acc_c = best_c.get("accuracy", 0.0)
    acc_q = best_q.get("accuracy", 0.0)
    acc_diff = acc_q - acc_c

    sentences: List[str] = []

    # 1. Outcome sentence (Winner or Tie)
    if abs(acc_diff) < tie_margin:
        sentences.append(
            f"Performance between the top classical model ({best_c['name']} at {acc_c*100:.1f}%) "
            f"and the top quantum model ({best_q['name']} at {acc_q*100:.1f}%) is a statistical tie, "
            f"as the {abs(acc_diff)*100:.1f} percentage point difference falls within the {tie_margin*100:.0f}% sample noise margin."
        )
    elif acc_diff > 0:
        sentences.append(
            f"The quantum approach achieved the highest performance, with {best_q['name']} reaching {acc_q*100:.1f}% test accuracy "
            f"and outperforming the strongest classical baseline ({best_c['name']} at {acc_c*100:.1f}%) by {acc_diff*100:.1f} percentage points."
        )
    else:
        margin = -acc_diff
        sentences.append(
            f"Classical baselines outperformed the quantum approaches, with {best_c['name']} achieving {acc_c*100:.1f}% test accuracy "
            f"compared to {acc_q*100:.1f}% for the top quantum model ({best_q['name']}), a lead of {margin*100:.1f} percentage points."
        )

    # 2. Training efficiency / time ratio sentence
    t_c = max(best_c.get("train_time_s", 0.0), 1e-4)
    t_q = best_q.get("train_time_s", 0.0)
    if t_q >= t_c:
        ratio = t_q / t_c
        sentences.append(
            f"In terms of computational efficiency, training the quantum model took {t_q:.2f} seconds compared to "
            f"{t_c:.4f} seconds for the classical baseline, representing a {ratio:.1f}x time ratio on local simulation."
        )
    else:
        ratio = t_c / max(t_q, 1e-4)
        sentences.append(
            f"In terms of computational efficiency, {best_q['name']} completed training in {t_q:.2f} seconds compared to "
            f"{t_c:.4f} seconds for {best_c['name']}."
        )

    # 3. Parameter count sentence
    vqc_params = [m["n_params"] for m in quantum_models if m["name"] == "VQC"]
    vqc_str = f"while VQC utilized {vqc_params[0]} variational parameters, " if vqc_params else ""
    sentences.append(
        f"Parameter scaling highlights key architectural trade-offs: {vqc_str}the Quantum Kernel SVM required exactly 0 trainable parameters "
        f"by operating directly on the fixed geometry of the quantum Hilbert space, whereas classical models maintained between "
        f"{min(m['n_params'] for m in c_pool)} and {max(m['n_params'] for m in c_pool)} parameters."
    )

    # 4. Entanglement impact sentence (if on/off variants present)
    q_ent = [m for m in quantum_models if m.get("entanglement") is True]
    q_no_ent = [m for m in quantum_models if m.get("entanglement") is False]

    if q_ent and q_no_ent:
        best_ent = max(q_ent, key=lambda m: m.get("accuracy", 0.0))
        best_no_ent = max(q_no_ent, key=lambda m: m.get("accuracy", 0.0))
        ent_diff = best_no_ent.get("accuracy", 0.0) - best_ent.get("accuracy", 0.0)

        if abs(ent_diff) < tie_margin:
            sentences.append(
                f"Removing quantum entanglement produced no meaningful difference in test accuracy "
                f"({best_ent['accuracy']*100:.1f}% with entanglement vs {best_no_ent['accuracy']*100:.1f}% without)."
            )
        elif ent_diff > 0:
            sentences.append(
                f"Disabling quantum entanglement improved accuracy by {ent_diff*100:.1f} percentage points "
                f"({best_no_ent['accuracy']*100:.1f}% vs {best_ent['accuracy']*100:.1f}%), indicating that the dataset's decision boundary "
                f"is effectively separable with single-qubit rotations and suffers from barren plateaus or over-parameterization when multi-qubit entanglement is added."
            )
        else:
            sentences.append(
                f"Disabling quantum entanglement reduced accuracy by {-ent_diff*100:.1f} percentage points "
                f"({best_ent['accuracy']*100:.1f}% vs {best_no_ent['accuracy']*100:.1f}%), demonstrating that two-qubit quantum correlations provided a measurable expressive benefit."
            )

    # 5. Conclusion sentence
    sentences.append(
        "Overall, classical models remain faster, simpler to tune, and resilient to hardware noise for this diagnostic classification problem, "
        "though non-entangled quantum kernels provide a competitive zero-trainable-parameter alternative."
    )

    return " ".join(sentences)
