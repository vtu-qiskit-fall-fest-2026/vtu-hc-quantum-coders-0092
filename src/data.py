"""Data loading and preprocessing utilities."""

from typing import Any, Tuple
import numpy as np
import pandas as pd


def load_sample_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a built-in toy or benchmark sample dataset by name.

    Args:
        name: Name of dataset (e.g., 'iris', 'breast_cancer', 'wine').

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels array.
    """
    # TODO: Implement dataset loading via sklearn.datasets or local sample files
    raise NotImplementedError("TODO: Implement load_sample_dataset")


def load_csv(file: Any) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from a CSV filepath or file-like buffer.

    Args:
        file: Path to CSV file or file-like buffer from file uploader.

    Returns:
        Tuple of (features_df, target_series).
    """
    # TODO: Implement CSV reading and feature/label extraction
    raise NotImplementedError("TODO: Implement load_csv")


def preprocess_and_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_size: float = 0.2,
    max_features: int = 8,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale features, reduce dimensionality to <=8 features (e.g. PCA), and train-test split.

    Args:
        X: Input feature matrix.
        y: Target labels.
        test_size: Proportion of dataset to include in test split.
        max_features: Maximum number of features to retain (<= 8 for quantum embedding).
        random_state: Fixed seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Implement standard scaling, PCA reduction to <= max_features, and train_test_split with fixed random_state
    raise NotImplementedError("TODO: Implement preprocess_and_split")
