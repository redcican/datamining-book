"""
Shared data loader for Breast Cancer Wisconsin dataset (§13.3).

Uses sklearn built-in dataset — no download needed.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def load_cancer() -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Load Breast Cancer Wisconsin and return (X_df, y_series, feature_names).

    Target: 0 = malignant (恶性), 1 = benign (良性).
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, list(data.feature_names)


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    """
    Return X_train, X_test, y_train, y_test, scaler, feature_names.

    All features are numerical, so only StandardScaler is needed.
    """
    X, y, feature_names = load_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()

    return X_train, X_test, y_train, y_test, scaler, feature_names
