"""
Shared data loader for MovieLens 100K dataset (§13.5).

Downloads from GroupLens and caches locally.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


_DATA_DIR = Path(__file__).resolve().parent / "data"
_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def _download_if_needed():
    """Download and extract MovieLens 100K if not already cached."""
    ml_dir = _DATA_DIR / "ml-100k"
    if (ml_dir / "u.data").exists():
        return ml_dir

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _DATA_DIR / "ml-100k.zip"

    if not zip_path.exists():
        print(f"Downloading MovieLens 100K from {_ZIP_URL} ...")
        urllib.request.urlretrieve(_ZIP_URL, zip_path)
        print(f"  Saved → {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(_DATA_DIR)
    print(f"  Extracted → {ml_dir}")

    return ml_dir


def load_movielens():
    """
    Load MovieLens 100K and return (ratings_df, movies_df).

    ratings_df columns: user_id, item_id, rating, timestamp
    movies_df columns:  item_id, title, genres (list)
    """
    ml_dir = _download_if_needed()

    # Ratings
    ratings = pd.read_csv(
        ml_dir / "u.data", sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1")

    # Movies
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"]
    movie_cols = (["item_id", "title", "release_date", "video_date", "url"]
                  + genre_names)
    movies = pd.read_csv(
        ml_dir / "u.item", sep="|", names=movie_cols,
        encoding="latin-1", usecols=["item_id", "title"] + genre_names)

    # Convert genre columns to a list
    movies["genres"] = movies[genre_names].apply(
        lambda row: [g for g, v in zip(genre_names, row) if v == 1],
        axis=1)
    movies = movies[["item_id", "title", "genres"]]

    return ratings, movies


def prepare_data(test_ratio=0.2, random_state=42):
    """
    Return train_df, test_df, n_users, n_items.

    Split: for each user, randomly hold out `test_ratio` of their ratings.
    user_id and item_id are 0-indexed in the returned DataFrames.
    """
    ratings, movies = load_movielens()

    # 0-index
    ratings = ratings.copy()
    ratings["user_id"] = ratings["user_id"] - 1
    ratings["item_id"] = ratings["item_id"] - 1
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()

    # Per-user random split
    rng = np.random.RandomState(random_state)
    train_idx, test_idx = [], []
    for uid, group in ratings.groupby("user_id"):
        indices = group.index.tolist()
        rng.shuffle(indices)
        split = max(1, int(len(indices) * test_ratio))
        test_idx.extend(indices[:split])
        train_idx.extend(indices[split:])

    train_df = ratings.loc[train_idx].reset_index(drop=True)
    test_df = ratings.loc[test_idx].reset_index(drop=True)

    return train_df, test_df, n_users, n_items, movies
