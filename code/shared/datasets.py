"""
Common dataset loaders used across textbook figure scripts.

All loaders return (X, y, feature_names, target_names) or similar tuples
so that figure scripts are self-contained.
"""
from sklearn import datasets
import numpy as np


def load_iris():
    """Iris — 150 samples, 4 features, 3 classes."""
    d = datasets.load_iris()
    return d.data, d.target, d.feature_names, d.target_names


def load_breast_cancer():
    """Breast Cancer Wisconsin — 569 samples, 30 features, binary."""
    d = datasets.load_breast_cancer()
    return d.data, d.target, d.feature_names, d.target_names


def load_digits():
    """Digits (MNIST-lite) — 1797 samples, 64 features, 10 classes."""
    d = datasets.load_digits()
    return d.data, d.target, d.feature_names, d.target_names


def make_blobs(n=300, centers=3, seed=42):
    """Simple isotropic Gaussian blobs — good for clustering demos."""
    X, y = datasets.make_blobs(n_samples=n, centers=centers, random_state=seed)
    return X, y


def make_moons(n=300, noise=0.08, seed=42):
    """Two interleaved half-moons — good for DBSCAN/SVM demos."""
    X, y = datasets.make_moons(n_samples=n, noise=noise, random_state=seed)
    return X, y


def make_circles(n=300, noise=0.05, seed=42):
    """Concentric circles — shows K-means failure modes."""
    X, y = datasets.make_circles(n_samples=n, noise=noise, random_state=seed)
    return X, y


def make_classification_hard(n=500, seed=42):
    """High-overlap binary classification — tests classifier limits."""
    X, y = datasets.make_classification(
        n_samples=n, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1,
        flip_y=0.1, random_state=seed,
    )
    return X, y


def synthetic_time_series(n=200, seed=42):
    """Simple synthetic time series with trend + seasonality + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 24)
    noise = rng.normal(0, 0.5, n)
    return t, trend + seasonal + noise
