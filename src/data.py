"""Data loading, cleaning, and preprocessing for the Breast Cancer Wisconsin benchmark."""

from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_breast_cancer_csv(
    path: str | Path = "data/breast-cancer-wisconsin-data.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and clean the Breast Cancer Wisconsin Diagnostic dataset.

    Drops the 'id' column and any stray 'Unnamed' columns.
    Encodes 'diagnosis' as 1 = malignant (M) and 0 = benign (B).

    Args:
        path: Filepath to the CSV dataset.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series) where:
            - X contains the 30 numeric feature columns.
            - y contains binary labels (1 = malignant, 0 = benign).
    """
    file_path = Path(path)
    if not file_path.exists():
        # Fallback to project root relative path
        root_path = Path(__file__).resolve().parent.parent / path
        if root_path.exists():
            file_path = root_path

    df = pd.read_csv(file_path)

    # Drop id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Drop any stray "Unnamed" columns
    unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Convention: 1 = malignant (M), 0 = benign (B).
    # This convention is used by every model and the verdict text.
    y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    X = df.drop(columns=["diagnosis"])

    return X, y


def preprocess_and_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_features: int = 4,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Preprocess dataset and split into train and test sets without data leakage.

    Steps:
    1. Stratified train/test split with fixed random_state.
    2. Fit StandardScaler on TRAIN set only, then transform both train and test.
    3. Fit PCA(n_components=n_features) on scaled TRAIN set only, transform both.
    4. Rescale PCA outputs with MinMaxScaler to [0, pi], fit on TRAIN only, applied to both
       (quantum feature maps work best with angle-sized inputs).

    Every model (classical and quantum) must use this exact dict so the comparison is fair.

    Args:
        X: Feature matrix (DataFrame or numpy array).
        y: Target labels (Series or numpy array).
        n_features: Number of principal components / features to retain.
        test_size: Proportion of dataset for test split (default: 0.25).
        random_state: Fixed random seed for reproducible split.

    Returns:
        Dict containing:
            - 'X_train': np.ndarray of shape (n_train, n_features) in [0, pi]
            - 'X_test': np.ndarray of shape (n_test, n_features)
            - 'y_train': np.ndarray of shape (n_train,)
            - 'y_test': np.ndarray of shape (n_test,)
            - 'n_features': int
            - 'explained_variance': float (sum of PCA explained variance ratio)
    """
    # 1. Stratified train/test split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Fit StandardScaler on TRAIN set only, transform both (no leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 3. Fit PCA on scaled TRAIN set only, transform both
    pca = PCA(n_components=n_features, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    explained_variance = float(np.sum(pca.explained_variance_ratio_))

    # 4. Rescale PCA outputs with MinMaxScaler to [0, pi], fit on TRAIN only
    minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_train_final = minmax.fit_transform(X_train_pca)
    X_test_final = minmax.transform(X_test_pca)

    return {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "y_train": np.asarray(y_train_raw),
        "y_test": np.asarray(y_test_raw),
        "n_features": n_features,
        "explained_variance": explained_variance,
    }


# Optional backward-compatibility helpers
def load_sample_dataset(name: str = "breast_cancer") -> Tuple[pd.DataFrame, pd.Series]:
    """Helper to load built-in sample dataset."""
    return load_breast_cancer_csv()


def load_csv(file: Any) -> Tuple[pd.DataFrame, pd.Series]:
    """Helper to load from a custom CSV path or buffer."""
    return load_breast_cancer_csv(file)


if __name__ == "__main__":
    X, y = load_breast_cancer_csv()
    data = preprocess_and_split(X, y, n_features=4)

    print(f"X shape: {X.shape}")
    print(f"Class counts:\n{y.value_counts().to_dict()}")
    print(f"Train shape: {data['X_train'].shape}, Test shape: {data['X_test'].shape}")
    print(f"Explained variance: {data['explained_variance']:.4f}")
    print(f"X_train min/max: [{data['X_train'].min():.4f}, {data['X_train'].max():.4f}]")
    print(f"X_test min/max: [{data['X_test'].min():.4f}, {data['X_test'].max():.4f}]")
