"""
Shared data loader for AI4I 2020 Predictive Maintenance dataset (§13.6).

Downloads from UCI ML Repository and caches locally.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


_DATA_DIR = Path(__file__).resolve().parent / "data"
_CSV_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00601/ai4i2020.csv")


def load_maintenance():
    """
    Load AI4I 2020 Predictive Maintenance dataset.

    Returns DataFrame with columns:
        Type, Air temperature [K], Process temperature [K],
        Rotational speed [rpm], Torque [Nm], Tool wear [min],
        Machine failure, TWF, HDF, PWF, OSF, RNF
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _DATA_DIR / "ai4i2020.csv"

    if not csv_path.exists():
        print(f"Downloading AI4I dataset from UCI ...")
        urllib.request.urlretrieve(_CSV_URL, csv_path)
        print(f"  Saved → {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop UDI and Product ID (not useful features)
    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def prepare_data(test_size=0.2, random_state=42):
    """
    Return X_train, X_test, y_train, y_test, preprocessor, feature_names, df.

    Target: 'Machine failure' (binary).
    Features: Type (categorical) + 5 numeric sensor readings.
    """
    df = load_maintenance()

    target = "Machine failure"
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    num_features = ["Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]",
                    "Tool wear [min]"]
    cat_features = ["Type"]

    X = df[cat_features + num_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False),
         cat_features),
    ])

    return (X_train, X_test, y_train, y_test, preprocessor,
            num_features, cat_features, df)
