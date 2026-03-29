"""
Shared data loader for Beijing PM2.5 dataset (§13.7).

Downloads from UCI ML Repository and caches locally.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


_DATA_DIR = Path(__file__).resolve().parent / "data"
_CSV_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00381/PRSA_data_2010.1.1-2014.12.31.csv")


def load_pm25():
    """
    Load Beijing PM2.5 dataset and return cleaned DataFrame with datetime index.

    Drops rows with missing PM2.5, adds datetime column.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _DATA_DIR / "PRSA_data.csv"

    if not csv_path.exists():
        print(f"Downloading Beijing PM2.5 dataset ...")
        urllib.request.urlretrieve(_CSV_URL, csv_path)
        print(f"  Saved → {csv_path}")

    df = pd.read_csv(csv_path)

    # Build datetime
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]])
    df = df.set_index("datetime").sort_index()

    # Drop rows with missing PM2.5
    df = df.dropna(subset=["pm2.5"])

    return df


def prepare_data():
    """
    Return X_train, X_test, y_train, y_test, feature_names, df_full.

    Feature engineering: lag features + rolling stats + temporal + weather.
    Split: train=2010-2013, test=2014 (temporal).
    """
    df = load_pm25()

    # ── Feature engineering ─────────────────────────────────
    df = df.copy()

    # Lag features
    df["pm25_lag1"] = df["pm2.5"].shift(1)
    df["pm25_lag24"] = df["pm2.5"].shift(24)

    # Rolling statistics
    df["pm25_roll24_mean"] = df["pm2.5"].shift(1).rolling(24).mean()
    df["pm25_roll24_std"] = df["pm2.5"].shift(1).rolling(24).std()

    # Temporal features
    df["hour"] = df.index.hour
    df["month"] = df.index.month

    # One-hot encode wind direction
    cbwd_dummies = pd.get_dummies(df["cbwd"], prefix="wind", dtype=float)
    df = pd.concat([df, cbwd_dummies], axis=1)

    # Drop NaN rows from lag/rolling
    df = df.dropna()

    # Feature columns
    weather_features = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    lag_features = ["pm25_lag1", "pm25_lag24",
                    "pm25_roll24_mean", "pm25_roll24_std"]
    temporal_features = ["hour", "month"]
    wind_cols = [c for c in df.columns if c.startswith("wind_")]

    feature_names = (lag_features + weather_features
                     + temporal_features + wind_cols)

    X = df[feature_names]
    y = df["pm2.5"]

    # Temporal split
    train_mask = df.index.year < 2014
    test_mask = df.index.year == 2014

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test, feature_names, df
