"""Classical baseline machine learning models for breast cancer classification."""

import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_classical_models(data: Dict[str, Any], random_state: int = 42) -> List[Dict[str, Any]]:
    """Train classical baseline classifiers (SVM, Random Forest, MLP) on preprocessed data.

    Args:
        data: Preprocessed dictionary containing 'X_train', 'X_test', 'y_train',
              'y_test', and 'n_features' (from src.data.preprocess_and_split).
        random_state: Random seed for reproducibility.

    Returns:
        List of dictionaries with exact keys:
            - 'name': str
            - 'type': 'classical'
            - 'accuracy': float
            - 'f1': float (for positive class 1 = malignant)
            - 'train_time_s': float (time.perf_counter around fit only)
            - 'n_params': int
            - 'predictions': List[int]
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    n_features = data.get("n_features", X_train.shape[1])

    results: List[Dict[str, Any]] = []

    # 1. SVM (sklearn SVC, RBF kernel)
    svm = SVC(kernel="rbf", random_state=random_state)
    t0 = time.perf_counter()
    svm.fit(X_train, y_train)
    t_svm = time.perf_counter() - t0
    preds_svm = svm.predict(X_test)
    # SVM size: number of stored support vectors multiplied by feature dimensionality
    n_params_svm = int(len(svm.support_) * n_features)
    results.append({
        "name": "SVM",
        "type": "classical",
        "accuracy": float(accuracy_score(y_test, preds_svm)),
        "f1": float(f1_score(y_test, preds_svm, pos_label=1)),
        "train_time_s": float(t_svm),
        "n_params": n_params_svm,
        "predictions": [int(p) for p in preds_svm],
    })

    # 2. Random Forest (100 trees)
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    t_rf = time.perf_counter() - t0
    preds_rf = rf.predict(X_test)
    # Random Forest size: total number of decision tree nodes across all 100 estimators
    n_params_rf = int(sum(tree.tree_.node_count for tree in rf.estimators_))
    results.append({
        "name": "Random Forest",
        "type": "classical",
        "accuracy": float(accuracy_score(y_test, preds_rf)),
        "f1": float(f1_score(y_test, preds_rf, pos_label=1)),
        "train_time_s": float(t_rf),
        "n_params": n_params_rf,
        "predictions": [int(p) for p in preds_rf],
    })

    # 3. MLP (sklearn MLPClassifier, one hidden layer of 8 units, max_iter=2000)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=random_state)
    t0 = time.perf_counter()
    mlp.fit(X_train, y_train)
    t_mlp = time.perf_counter() - t0
    preds_mlp = mlp.predict(X_test)
    # MLP size: exact count of trainable connection weights plus layer bias terms
    n_params_mlp = int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
    results.append({
        "name": "MLP",
        "type": "classical",
        "accuracy": float(accuracy_score(y_test, preds_mlp)),
        "f1": float(f1_score(y_test, preds_mlp, pos_label=1)),
        "train_time_s": float(t_mlp),
        "n_params": n_params_mlp,
        "predictions": [int(p) for p in preds_mlp],
    })

    return results


def run_classical_on_full_features(
    X_train_full: np.ndarray | pd.DataFrame,
    X_test_full: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    y_test: np.ndarray | pd.Series,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Train classical baseline classifiers on all 30 features (StandardScaler fit on train only).

    Args:
        X_train_full: Raw unreduced training features.
        X_test_full: Raw unreduced testing features.
        y_train: Training labels.
        y_test: Testing labels.
        random_state: Random seed for reproducibility.

    Returns:
        List of dictionaries with identical format, labelled with type='classical_full'.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_full)
    X_te_scaled = scaler.transform(X_test_full)
    n_features = X_tr_scaled.shape[1]
    y_tr = np.asarray(y_train)
    y_te = np.asarray(y_test)

    results: List[Dict[str, Any]] = []

    # 1. SVM on full features
    svm = SVC(kernel="rbf", random_state=random_state)
    t0 = time.perf_counter()
    svm.fit(X_tr_scaled, y_tr)
    t_svm = time.perf_counter() - t0
    preds_svm = svm.predict(X_te_scaled)
    # SVM size: number of stored support vectors multiplied by feature dimensionality
    n_params_svm = int(len(svm.support_) * n_features)
    results.append({
        "name": "SVM",
        "type": "classical_full",
        "accuracy": float(accuracy_score(y_te, preds_svm)),
        "f1": float(f1_score(y_te, preds_svm, pos_label=1)),
        "train_time_s": float(t_svm),
        "n_params": n_params_svm,
        "predictions": [int(p) for p in preds_svm],
    })

    # 2. Random Forest on full features
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    t0 = time.perf_counter()
    rf.fit(X_tr_scaled, y_tr)
    t_rf = time.perf_counter() - t0
    preds_rf = rf.predict(X_te_scaled)
    # Random Forest size: total number of decision tree nodes across all 100 estimators
    n_params_rf = int(sum(tree.tree_.node_count for tree in rf.estimators_))
    results.append({
        "name": "Random Forest",
        "type": "classical_full",
        "accuracy": float(accuracy_score(y_te, preds_rf)),
        "f1": float(f1_score(y_te, preds_rf, pos_label=1)),
        "train_time_s": float(t_rf),
        "n_params": n_params_rf,
        "predictions": [int(p) for p in preds_rf],
    })

    # 3. MLP on full features
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=random_state)
    t0 = time.perf_counter()
    mlp.fit(X_tr_scaled, y_tr)
    t_mlp = time.perf_counter() - t0
    preds_mlp = mlp.predict(X_te_scaled)
    # MLP size: exact count of trainable connection weights plus layer bias terms
    n_params_mlp = int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
    results.append({
        "name": "MLP",
        "type": "classical_full",
        "accuracy": float(accuracy_score(y_te, preds_mlp)),
        "f1": float(f1_score(y_te, preds_mlp, pos_label=1)),
        "train_time_s": float(t_mlp),
        "n_params": n_params_mlp,
        "predictions": [int(p) for p in preds_mlp],
    })

    return results


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data import load_breast_cancer_csv, preprocess_and_split

    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    # 1. Models on reduced 4 features (PCA + MinMaxScaler [0, pi])
    reduced_results = train_classical_models(data)

    # 2. Reference models on full 30 features
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    full_results = run_classical_on_full_features(X_train_full, X_test_full, y_train, y_test)

    all_results = reduced_results + full_results

    # Print neat table
    header = f"{'Name':<15} {'Type':<16} {'Accuracy':<10} {'F1':<10} {'Train Time (s)':<16} {'Params':<10}"
    print(header)
    print("-" * len(header))
    for res in all_results:
        print(
            f"{res['name']:<15} {res['type']:<16} {res['accuracy']:<10.4f} "
            f"{res['f1']:<10.4f} {res['train_time_s']:<16.4f} {res['n_params']:<10}"
        )
