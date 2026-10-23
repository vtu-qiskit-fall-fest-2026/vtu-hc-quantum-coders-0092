"""Streamlit Web Application: Quantum vs Classical ML Benchmark."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.classical import train_classical_models
from src.data import load_breast_cancer_csv, preprocess_and_split
from src.quantum import save_circuit_diagrams, train_quantum_kernel_svm, train_vqc
from src.verdict import build_verdict

# Optional modules with graceful fallback
try:
    from src.noise import run_noise_sweep
except Exception:
    run_noise_sweep = None

try:
    from src.study import run_entanglement_study
except Exception:
    run_entanglement_study = None

RESULTS_DIR = Path("results")
LAST_RUN_FILE = RESULTS_DIR / "last_run.json"


def serialize_model_results(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip private/non-serializable objects from model dicts for JSON export."""
    clean = []
    for m in models:
        c = {k: v for k, v in m.items() if not k.startswith("_")}
        clean.append(c)
    return clean


def save_results(data_dict: Dict[str, Any]) -> None:
    """Save benchmark run results to results/last_run.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2)


def load_saved_results() -> Optional[Dict[str, Any]]:
    """Load benchmark run results from results/last_run.json if present."""
    if LAST_RUN_FILE.exists():
        with open(LAST_RUN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main() -> None:
    st.set_page_config(
        page_title="Quantum vs Classical ML Benchmark",
        page_icon="⚛️",
        layout="wide",
    )

    st.title("⚛️ Quantum vs Classical ML Benchmark")
    st.caption("A rigorous, head-to-head empirical comparison of Qiskit quantum classifiers and classical baselines on identical data.")
    st.markdown(
        "<small style='color: gray;'>Dataset Credit: Breast Cancer Wisconsin (Diagnostic) — "
        "W. Wolberg, O. Mangasarian, N. Street, UCI Machine Learning Repository (CC BY 4.0).</small>",
        unsafe_allow_html=True,
    )

    # Initialize session state keys: auto-load from results/last_run.json if present
    if "results" not in st.session_state or st.session_state["results"] is None:
        saved = load_saved_results()
        if saved:
            st.session_state["results"] = saved.get("results")
            st.session_state["noise_results"] = saved.get("noise_results")
            st.session_state["study_results"] = saved.get("study_results")
            st.session_state["verdict"] = saved.get("verdict")
            st.session_state["data_source_info"] = "Loaded from cached benchmark results (results/last_run.json)."
        else:
            st.session_state["results"] = None
            st.session_state["noise_results"] = None
            st.session_state["study_results"] = None
            st.session_state["verdict"] = None
            st.session_state["data_source_info"] = None

    # Sidebar Configuration
    st.sidebar.header("⚙️ Benchmark Settings")

    dataset_choice = st.sidebar.radio(
        "Dataset Source",
        ["Built-in Breast Cancer Wisconsin CSV", "Upload CSV"],
    )

    uploaded_file = None
    label_col = None
    pos_class = None

    if dataset_choice == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
        if uploaded_file is not None:
            preview_df = pd.read_csv(uploaded_file)
            st.sidebar.write(f"Columns found: {len(preview_df.columns)}")
            label_col = st.sidebar.selectbox("Select Target / Label Column", preview_df.columns)
            unique_vals = list(preview_df[label_col].dropna().unique())
            pos_class = st.sidebar.selectbox("Select Positive Class Value", unique_vals)
            uploaded_file.seek(0)

    exec_profile = st.sidebar.selectbox(
        "Execution Profile",
        ["Fast Interactive (~1-2 min)", "Full Research (~4-6 min)"],
        index=0,
        help="Fast Interactive uses n_train=60 and maxiter=50 for quick responsiveness. Full Research uses n_train=120 and maxiter=100.",
    )

    default_ntrain = 60 if "Fast" in exec_profile else 120
    default_maxiter = 50 if "Fast" in exec_profile else 100

    vqc_maxiter = st.sidebar.slider("VQC Max Iterations (COBYLA)", min_value=20, max_value=200, value=default_maxiter, step=10)
    n_train = st.sidebar.slider("Training Samples Subsample (n_train)", min_value=30, max_value=150, value=default_ntrain, step=10)

    st.sidebar.subheader("Additional Stages")
    include_noise = st.sidebar.checkbox("Include Noise Sweep (Stage 4)", value=True, help="Fast sweep across 5 depolarizing noise levels and IBM FakeTorino.")
    include_study = st.sidebar.checkbox("Include Multi-Seed Study (Stage 5)", value=False, help="Runs multi-seed iterations; adds ~3-4 minutes. If unchecked, preserves cached study results.")

    st.sidebar.divider()
    run_btn = st.sidebar.button("🚀 Run Benchmark", type="primary", use_container_width=True)
    load_btn = st.sidebar.button("📂 Reload Saved Results", use_container_width=True)

    # Action: Reload Saved Results
    if load_btn:
        saved = load_saved_results()
        if saved:
            st.session_state["results"] = saved.get("results")
            st.session_state["noise_results"] = saved.get("noise_results")
            st.session_state["study_results"] = saved.get("study_results")
            st.session_state["verdict"] = saved.get("verdict")
            st.session_state["data_source_info"] = "Reloaded from results/last_run.json."
            st.success("Successfully loaded saved benchmark results from results/last_run.json")
        else:
            st.warning("No saved results found at results/last_run.json. Run a benchmark first.")

    # Action: Run Benchmark
    if run_btn:
        with st.status("Running Quantum vs Classical ML Benchmark...", expanded=True) as status:
            data = None
            all_models = []
            noise_res = st.session_state.get("noise_results")
            study_res = st.session_state.get("study_results")

            # Stage 1: Data Preparation
            try:
                status.write("📊 Stage 1/5: Loading and preprocessing dataset...")
                if dataset_choice == "Upload CSV" and uploaded_file is not None and label_col is not None:
                    raw_df = pd.read_csv(uploaded_file)
                    y_raw = (raw_df[label_col] == pos_class).astype(int)
                    num_cols = raw_df.select_dtypes(include=[np.number]).columns.drop(label_col, errors="ignore")
                    X_raw = raw_df[num_cols].fillna(0)
                    data = preprocess_and_split(X_raw, y_raw, n_features=4)
                else:
                    X, y = load_breast_cancer_csv()
                    data = preprocess_and_split(X, y, n_features=4)
                status.write(f"✓ Data ready: {data['X_train'].shape[0]} train, {data['X_test'].shape[0]} test samples.")
            except Exception as e:
                st.error(f"Data preparation error: {e}")

            # Stage 2: Classical Baselines
            if data is not None:
                try:
                    status.write("💻 Stage 2/5: Training classical baseline models (SVM, Random Forest, MLP)...")
                    c_results = train_classical_models(data)
                    all_models.extend(c_results)
                    status.write("✓ Classical models evaluated.")
                except Exception as e:
                    st.error(f"Classical modeling error: {e}")

            # Stage 3: Quantum Models (4 combinations)
            vqc_ent = None
            vqc_no_ent = None
            svm_ent = None
            svm_no_ent = None

            if data is not None:
                try:
                    status.write(f"⚛️ Stage 3/5: Training VQC (Entangled, n_train={n_train}, maxiter={vqc_maxiter})...")
                    vqc_ent = train_vqc(data, entanglement=True, maxiter=vqc_maxiter, n_train=n_train)
                    all_models.append(vqc_ent)

                    status.write(f"⚛️ Stage 3/5: Training VQC (No Entanglement, n_train={n_train}, maxiter={vqc_maxiter})...")
                    vqc_no_ent = train_vqc(data, entanglement=False, maxiter=vqc_maxiter, n_train=n_train)
                    all_models.append(vqc_no_ent)

                    status.write(f"⚛️ Stage 3/5: Training Quantum Kernel SVM (Entangled, n_train={n_train})...")
                    svm_ent = train_quantum_kernel_svm(data, entanglement=True, n_train=n_train)
                    all_models.append(svm_ent)

                    status.write(f"⚛️ Stage 3/5: Training Quantum Kernel SVM (No Entanglement, n_train={n_train})...")
                    svm_no_ent = train_quantum_kernel_svm(data, entanglement=False, n_train=n_train)
                    all_models.append(svm_no_ent)
                    status.write("✓ All 4 quantum models evaluated.")
                except Exception as e:
                    st.error(f"Quantum modeling error: {e}")

            # Stage 4: Noise Sweep
            if data is not None and run_noise_sweep is not None and include_noise:
                try:
                    status.write("🌪️ Stage 4/5: Running noise sweep across quantum models...")
                    q_models_to_test = [m for m in [vqc_ent, vqc_no_ent, svm_ent, svm_no_ent] if m is not None]
                    noise_res = []
                    for qm in q_models_to_test:
                        # Quick responsive evaluation on 20 test points with 128 shots
                        sweep = run_noise_sweep(data, qm, shots=128, n_test=20)
                        noise_res.extend(sweep)
                    status.write("✓ Noise sweep completed.")
                except Exception as e:
                    st.warning(f"Noise sweep skipped or encountered an issue: {e}")
            else:
                if not include_noise:
                    status.write("⏭️ Stage 4/5: Noise sweep skipped by user (retained cached data).")
                else:
                    status.write("ℹ️ Noise sweep module not available.")

            # Stage 5: Entanglement Robustness Study
            if run_entanglement_study is not None and include_study:
                try:
                    status.write("🔬 Stage 5/5: Running multi-seed entanglement study...")
                    study_res = run_entanglement_study(seeds=(0, 1), n_train=min(n_train, 60), maxiter_vqc=min(vqc_maxiter, 60))
                    status.write("✓ Entanglement study completed.")
                except Exception as e:
                    st.warning(f"Entanglement study skipped or encountered an issue: {e}")
            else:
                if not include_study:
                    status.write("⏭️ Stage 5/5: Entanglement study skipped by user (retained cached data).")
                else:
                    status.write("ℹ️ Entanglement study module not available.")

            # Verdict Generation
            try:
                verdict_text = build_verdict(all_models)
            except Exception as e:
                verdict_text = f"Verdict could not be generated: {e}"

            # Save in session state and JSON file
            st.session_state["results"] = serialize_model_results(all_models)
            st.session_state["noise_results"] = noise_res
            st.session_state["study_results"] = study_res
            st.session_state["verdict"] = verdict_text
            st.session_state["data_source_info"] = f"Live run completed at {time.strftime('%H:%M:%S')} (n_train={n_train}, maxiter={vqc_maxiter})."

            save_results({
                "results": st.session_state["results"],
                "noise_results": noise_res,
                "study_results": study_res,
                "verdict": verdict_text,
                "timestamp": time.time(),
            })

            status.update(label="Benchmark Complete! Results saved to results/last_run.json", state="complete")

    # Main Area Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🔬 Quantum Insights", "⚖️ Verdict", "ℹ️ About"])

    results = st.session_state.get("results")
    if st.session_state.get("data_source_info"):
        st.info(f"ℹ️ {st.session_state['data_source_info']}")

    # ---------------- TAB 1: RESULTS ----------------
    with tab1:
        st.subheader("Model Benchmark Comparison")
        if not results:
            st.info("No benchmark results to display. Click **'Run Benchmark'** in the sidebar or **'Load Last Saved Results'**.")
        else:
            table_rows = []
            for r in results:
                ent_val = str(r.get("entanglement", "—"))
                two_q = r.get("two_qubit_gate_count", "—")
                table_rows.append({
                    "Model Name": r["name"],
                    "Type": r["type"],
                    "Entanglement": ent_val,
                    "Accuracy": f"{r['accuracy']:.4f}",
                    "F1 Score": f"{r['f1']:.4f}",
                    "Train Time (s)": f"{r['train_time_s']:.4f}",
                    "Params": r["n_params"],
                    "2-Qubit Gates": two_q,
                })
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                # Accuracy Bar Chart
                plot_df = pd.DataFrame(results).copy()
                plot_df["Label"] = plot_df.apply(
                    lambda x: f"{x['name']} ({'Ent' if x.get('entanglement') is True else 'NoEnt' if x.get('entanglement') is False else 'Class'})",
                    axis=1,
                )
                fig_acc = px.bar(
                    plot_df,
                    x="Label",
                    y="accuracy",
                    color="type",
                    title="Test Accuracy Comparison",
                    labels={"accuracy": "Accuracy", "Label": "Model"},
                    text_auto=".3f",
                    color_discrete_map={"classical": "#1f77b4", "quantum": "#2ca02c", "classical_full": "#aec7e8"},
                )
                fig_acc.update_layout(yaxis_range=[0.5, 1.05])
                st.plotly_chart(fig_acc, use_container_width=True)

            with col2:
                # Training Time Bar Chart
                fig_time = px.bar(
                    plot_df,
                    x="Label",
                    y="train_time_s",
                    color="type",
                    title="Training Time (seconds, log scale)",
                    labels={"train_time_s": "Train Time (s)", "Label": "Model"},
                    log_y=True,
                    color_discrete_map={"classical": "#1f77b4", "quantum": "#2ca02c", "classical_full": "#aec7e8"},
                )
                st.plotly_chart(fig_time, use_container_width=True)

    # ---------------- TAB 2: QUANTUM INSIGHTS ----------------
    with tab2:
        st.subheader("Quantum Entanglement, Noise & Circuit Architecture")
        if not results:
            st.info("Run the benchmark or load saved results to view quantum insights.")
        else:
            q_models = [r for r in results if r.get("type") == "quantum"]
            if q_models:
                # Entanglement Comparison Chart
                ent_df = pd.DataFrame(q_models).copy()
                ent_df["Entanglement_Label"] = ent_df["entanglement"].map({True: "Entangled (ZZ)", False: "No Entanglement (Z)"})
                fig_ent = px.bar(
                    ent_df,
                    x="name",
                    y="accuracy",
                    color="Entanglement_Label",
                    barmode="group",
                    title="Impact of Entanglement on Test Accuracy",
                    labels={"accuracy": "Accuracy", "name": "Quantum Model Architecture"},
                    text_auto=".3f",
                )
                fig_ent.update_layout(yaxis_range=[0.5, 1.05])
                st.plotly_chart(fig_ent, use_container_width=True)

            # Noise Sweep Line Chart
            st.markdown("#### Robustness under Depolarizing Noise & Fake IBM Hardware")
            noise_res = st.session_state.get("noise_results")
            if noise_res:
                df_noise = pd.DataFrame(noise_res).copy()
                # Separate numerical depolarizing levels from fake_backend for clean plotting
                df_depol = df_noise[df_noise["noise_level"] != "fake_backend"].copy()
                df_depol["noise_level"] = df_depol["noise_level"].astype(float)
                df_depol["Model Variant"] = df_depol.apply(
                    lambda x: f"{x['model_name']} ({'Entangled' if x['entanglement'] else 'No Entanglement'})",
                    axis=1,
                )

                fig_noise = px.line(
                    df_depol,
                    x="noise_level",
                    y="accuracy",
                    color="Model Variant",
                    markers=True,
                    title="Accuracy vs Depolarizing Error Rate",
                    labels={"noise_level": "Single-Qubit Depolarizing Error (p)", "accuracy": "Test Accuracy"},
                )
                st.plotly_chart(fig_noise, use_container_width=True)

                # Show fake_backend point in a clean metric callout
                df_fake = df_noise[df_noise["noise_level"] == "fake_backend"]
                if not df_fake.empty:
                    st.write("**Performance under IBM Fake Torino Calibration Noise:**")
                    cols = st.columns(len(df_fake))
                    for col, (_, row) in zip(cols, df_fake.iterrows()):
                        var_str = "Entangled" if row["entanglement"] else "No Ent"
                        col.metric(f"{row['model_name']} ({var_str})", f"{row['accuracy']*100:.1f}%")
            else:
                st.caption("Noise sweep results not available yet.")

            # Multi-seed Entanglement Study Table
            st.markdown("#### Multi-Seed Entanglement Robustness Study")
            study_res = st.session_state.get("study_results")
            if study_res and "kernel_svm" in study_res:
                k_data = study_res["kernel_svm"]
                study_df = pd.DataFrame([
                    {
                        "Architecture": "Quantum Kernel SVM (Entangled ZZ)",
                        "Mean Accuracy": f"{k_data['entangled']['mean']:.4f}",
                        "Std Dev": f"{k_data['entangled']['std']:.4f}",
                    },
                    {
                        "Architecture": "Quantum Kernel SVM (Non-Entangled Z)",
                        "Mean Accuracy": f"{k_data['non_entangled']['mean']:.4f}",
                        "Std Dev": f"{k_data['non_entangled']['std']:.4f}",
                    },
                ])
                st.table(study_df)
            else:
                st.caption("Entanglement multi-seed study not available yet.")

            # Circuit Diagrams from docs/
            st.markdown("#### Quantum Circuit Diagrams")
            # Ensure diagrams exist
            docs_dir = Path("docs")
            if not (docs_dir / "feature_map_entangled.png").exists():
                try:
                    save_circuit_diagrams("docs/")
                except Exception:
                    pass

            c1, c2 = st.columns(2)
            with c1:
                fm_ent_p = docs_dir / "feature_map_entangled.png"
                if fm_ent_p.exists():
                    st.image(str(fm_ent_p), caption="Feature Map (Entangled ZZ)")
                ans_ent_p = docs_dir / "ansatz_entangled.png"
                if ans_ent_p.exists():
                    st.image(str(ans_ent_p), caption="RealAmplitudes Ansatz (Entangled)")

            with c2:
                fm_no_p = docs_dir / "feature_map_no_entanglement.png"
                if fm_no_p.exists():
                    st.image(str(fm_no_p), caption="Feature Map (No Entanglement Z)")
                ans_no_p = docs_dir / "ansatz_no_entanglement.png"
                if ans_no_p.exists():
                    st.image(str(ans_no_p), caption="RealAmplitudes Ansatz (No Entanglement)")

    # ---------------- TAB 3: VERDICT ----------------
    with tab3:
        st.subheader("Objective Scientific Verdict")
        verdict = st.session_state.get("verdict")
        if verdict:
            st.info(verdict)
        else:
            st.info("Run the benchmark or load saved results to generate an empirical verdict.")

        st.markdown("---")
        st.markdown("#### 📖 How to Read this Verdict")
        st.markdown(
            """
            - **Statistical Significance & Small Test Sets:** The benchmark test split contains 143 samples. 
              A single sample difference corresponds to approximately ~0.7 percentage points of accuracy. 
              Therefore, any accuracy difference smaller than **2.0 percentage points** is considered a **statistical tie**, 
              as it cannot be distinguished from finite-sample noise.
            - **Entanglement Benefit:** When non-entangled models match or exceed entangled models, it demonstrates that 
              pairwise quantum correlations are not needed to model the underlying feature relationships for this task.
            - **Zero Trainable Parameters:** The Quantum Kernel SVM does not tune variational weights; it maps inputs into a 
              fixed 4-qubit Hilbert space and delegates convex optimization to a classical dual SVM solver.
            """
        )

    # ---------------- TAB 4: ABOUT ----------------
    with tab4:
        st.subheader("About the Benchmark Suite")
        st.markdown(
            r"""
            This application provides a fair, reproducible benchmark comparing classical machine learning algorithms 
            (Support Vector Classifier with RBF kernel, Random Forest, and Multi-Layer Perceptrons) with Quantum Machine Learning 
            models (Variational Quantum Classifiers and Quantum Kernel Support Vector Classifiers). Both paradigms evaluate on 
            the exact same stratified train/test split, normalized to $[0, \pi]$ angles for quantum embedding.

            **Technology Stack:**
            - **Quantum Framework:** Qiskit 2.x, Qiskit Machine Learning, Qiskit Aer Simulator
            - **Noise Modeling:** Qiskit Aer Depolarizing Error Channels & IBM Heron (`FakeTorino`) calibration models
            - **Classical Baselines:** Scikit-Learn (SVM, RandomForest, MLP)
            - **User Interface & Analytics:** Streamlit & Plotly Express

            **Dataset Credit:**
            - **Dataset:** Breast Cancer Wisconsin (Diagnostic)
            - **Authors:** W. Wolberg, O. Mangasarian, N. Street (University of Wisconsin)
            - **Source:** UCI Machine Learning Repository, licensed under **CC BY 4.0**
            """
        )


if __name__ == "__main__":
    main()
