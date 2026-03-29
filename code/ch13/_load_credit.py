"""
Shared data loader for German Credit dataset (§13.2).

Downloads via sklearn's fetch_openml (auto-cached in ~/scikit_learn_data/).
Provides raw loading and train/test preparation with ColumnTransformer.
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


def load_credit() -> pd.DataFrame:
    """Load German Credit dataset from OpenML, return raw DataFrame."""
    credit = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    return credit.frame.copy()


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load, split, and return preprocessing pipeline + data.

    Returns:
        (X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols, df)
    """
    df = load_credit()

    # Target: 1 = bad credit (minority), 0 = good credit
    y = (df["class"] == "bad").astype(int)
    X = df.drop(columns=["class"])

    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False,
                              handle_unknown="ignore"), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols, df
