"""Classical machine learning baseline models."""

from typing import Any, Dict
import numpy as np


def train_classical_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Train classical baseline classifiers (SVM, Random Forest, MLP) on the split.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.

    Returns:
        Dictionary mapping model names to evaluation results, predictions, and metrics.
    """
    # TODO: Train SVC, RandomForestClassifier, MLPClassifier
    # TODO: Calculate accuracy, precision, recall, F1, fit time, and inference time
    raise NotImplementedError("TODO: Implement train_classical_models")
