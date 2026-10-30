"""Memory-aware loading and feature construction for headerless ratings CSV files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

COLUMNS = ["user_id", "product_id", "rating", "timestamp"]


def read_ratings(path: str | Path, chunksize: int = 250_000) -> Iterator[pd.DataFrame]:
    """Yield clean chunks from either a headerless or headered ratings export.

    The supplied data has no header and follows userId, productId, rating,
    timestamp. Timestamp is retained for optional analysis but not used in models.
    """
    path = Path(path)
    first = pd.read_csv(path, header=None, nrows=1, dtype="string")
    has_header = set(first.iloc[0].str.lower()) >= {"userid", "productid", "rating"}
    reader = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else COLUMNS,
        chunksize=chunksize,
        dtype={"user_id": "string", "product_id": "string"} if not has_header else None,
    )
    for chunk in reader:
        chunk.columns = [str(c).strip().lower().replace(" ", "_") for c in chunk.columns]
        aliases = {"userid": "user_id", "user_id": "user_id", "productid": "product_id", "product_id": "product_id"}
        chunk = chunk.rename(columns=aliases)
        required = {"user_id", "product_id", "rating"}
        if not required.issubset(chunk.columns):
            raise ValueError(f"Expected columns {sorted(required)}; found {list(chunk.columns)}")
        chunk = chunk[[c for c in COLUMNS if c in chunk.columns]].copy()
        chunk["user_id"] = chunk["user_id"].astype("string").str.strip()
        chunk["product_id"] = chunk["product_id"].astype("string").str.strip()
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk = chunk.dropna(subset=["user_id", "product_id", "rating"])
        chunk = chunk[chunk["rating"].between(1, 5)]
        yield chunk


def load_sample(path: str | Path, max_rows: int, chunksize: int = 250_000) -> pd.DataFrame:
    """Read up to max_rows deterministically without loading a large CSV at once."""
    parts, remaining = [], max_rows
    for chunk in read_ratings(path, chunksize):
        part = chunk.iloc[:remaining]
        parts.append(part)
        remaining -= len(part)
        if remaining <= 0:
            break
    if not parts:
        raise ValueError("No valid ratings were found.")
    return pd.concat(parts, ignore_index=True)


def user_features(ratings: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable per-user features used by each clustering model."""
    features = ratings.groupby("user_id").agg(
        rating_count=("rating", "size"),
        mean_rating=("rating", "mean"),
        rating_std=("rating", "std"),
        unique_products=("product_id", "nunique"),
    )
    features["rating_std"] = features["rating_std"].fillna(0.0)
    features["high_rating_share"] = ratings.assign(high=ratings["rating"] >= 4).groupby("user_id")["high"].mean()
    features["unique_product_ratio"] = features["unique_products"] / features["rating_count"]
    return features.reset_index()
