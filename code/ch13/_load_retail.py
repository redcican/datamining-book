"""
Shared data loader for UCI Online Retail dataset (§13.1).

Downloads the Excel file once, caches in code/ch13/data/, and provides
cleaned DataFrames for all figure/case scripts.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent / "data"
_CACHE_FILE = _DATA_DIR / "OnlineRetail.xlsx"
_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/00352/Online%20Retail.xlsx")


def load_raw() -> pd.DataFrame:
    """Download (if needed) and return the raw Online Retail DataFrame."""
    if not _CACHE_FILE.exists():
        print(f"Downloading dataset from UCI … (≈23 MB)")
        df = pd.read_excel(_URL)
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(_DATA_DIR / "online_retail_raw.pkl")
        # Also save xlsx for reference
        df.to_excel(_CACHE_FILE, index=False)
        print(f"Cached → {_CACHE_FILE}")
    else:
        pkl = _DATA_DIR / "online_retail_raw.pkl"
        if pkl.exists():
            df = pd.read_pickle(pkl)
        else:
            df = pd.read_excel(_CACHE_FILE)
            df.to_pickle(pkl)
    return df


def load_clean(country: str = "United Kingdom") -> tuple[pd.DataFrame, dict]:
    """
    Return cleaned DataFrame and a stats dict documenting the cleaning steps.

    Stats dict keys: raw_rows, raw_invoices, after_drop_na, after_drop_returns,
    after_positive_qty, after_country, clean_rows, clean_invoices.
    """
    df = load_raw()
    stats = {
        "raw_rows": len(df),
        "raw_invoices": df["InvoiceNo"].nunique(),
    }

    df = df.dropna(subset=["InvoiceNo", "Description"])
    stats["after_drop_na"] = len(df)

    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    stats["after_drop_returns"] = len(df)

    df = df[df["Quantity"] > 0]
    stats["after_positive_qty"] = len(df)

    if country:
        df = df[df["Country"] == country]
    stats["after_country"] = len(df)

    stats["clean_rows"] = len(df)
    stats["clean_invoices"] = df["InvoiceNo"].nunique()

    return df, stats


def build_basket(df: pd.DataFrame, min_support: float = 0.03):
    """
    Build transaction matrix and run Apriori.

    Returns (basket_df, freq_items_df, rules_df).
    """
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    transactions = (df.groupby("InvoiceNo")["Description"]
                    .apply(list).tolist())

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_)

    freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    freq_items["length"] = freq_items["itemsets"].apply(len)

    rules = association_rules(freq_items, metric="lift", min_threshold=1.0,
                              num_itemsets=len(freq_items))
    rules = rules.sort_values("lift", ascending=False)

    return basket, freq_items, rules, transactions
