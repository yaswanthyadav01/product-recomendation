"""Clustering comparison and item-item collaborative-filtering utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["rating_count", "mean_rating", "rating_std", "unique_products", "high_rating_share", "unique_product_ratio"]


@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray
    metrics: dict[str, float]


def compare_clusterers(features: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> tuple[pd.DataFrame, dict[str, ClusterResult], StandardScaler]:
    """Fit three complementary clustering methods and return comparable scores."""
    X = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models = {
        "K-Means": KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state),
        "MiniBatch K-Means": MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, n_init=10, random_state=random_state),
        "Gaussian Mixture": GaussianMixture(n_components=n_clusters, random_state=random_state),
    }
    results: dict[str, ClusterResult] = {}
    rows = []
    for name, model in models.items():
        labels = model.fit_predict(X_scaled)
        # Silhouette is costly on very large samples; a fixed sample remains comparable.
        sample = min(10_000, len(X_scaled))
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_scaled), sample, replace=False) if sample < len(X_scaled) else np.arange(len(X_scaled))
        metrics = {
            "silhouette": float(silhouette_score(X_scaled[idx], labels[idx])),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
        }
        results[name] = ClusterResult(name, labels, metrics)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False), results, scaler


def build_item_knn(ratings: pd.DataFrame, min_product_ratings: int = 5, min_user_ratings: int = 3):
    """Fit a cosine nearest-neighbor model over a sparse product-by-user matrix."""
    product_counts = ratings["product_id"].value_counts()
    user_counts = ratings["user_id"].value_counts()
    filtered = ratings[
        ratings["product_id"].isin(product_counts[product_counts >= min_product_ratings].index)
        & ratings["user_id"].isin(user_counts[user_counts >= min_user_ratings].index)
    ].copy()
    # Duplicate ratings for the same pair are collapsed to a mean before sparse conversion.
    filtered = filtered.groupby(["product_id", "user_id"], as_index=False)["rating"].mean()
    products = pd.Index(sorted(filtered["product_id"].unique()))
    users = pd.Index(sorted(filtered["user_id"].unique()))
    product_codes = products.get_indexer(filtered["product_id"])
    user_codes = users.get_indexer(filtered["user_id"])
    matrix = csr_matrix((filtered["rating"].to_numpy(), (product_codes, user_codes)), shape=(len(products), len(users)))
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(21, len(products)), n_jobs=-1)
    model.fit(matrix)
    popularity = ratings.groupby("product_id").agg(mean_rating=("rating", "mean"), rating_count=("rating", "size")).sort_values(["rating_count", "mean_rating"], ascending=False)
    return model, matrix, products, popularity


def similar_products(product_id: str, model, matrix: csr_matrix, products: pd.Index, popularity: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return nearest products, falling back to popular products for unseen IDs."""
    if product_id not in products:
        return popularity.head(n).reset_index().assign(similarity=np.nan)
    pos = products.get_loc(product_id)
    distances, indices = model.kneighbors(matrix[pos], n_neighbors=min(n + 1, len(products)))
    out = pd.DataFrame({"product_id": products[indices[0]], "similarity": 1 - distances[0]})
    out = out[out["product_id"] != product_id].head(n)
    return out.merge(popularity.reset_index(), on="product_id", how="left")


def recommendations_for_user(user_id: str, ratings: pd.DataFrame, cluster_labels: pd.DataFrame, popularity: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Recommend well-rated popular products within the user's behavior segment."""
    user_cluster = cluster_labels.loc[cluster_labels["user_id"] == user_id, "cluster"]
    seen = set(ratings.loc[ratings["user_id"] == user_id, "product_id"])
    if user_cluster.empty:
        candidates = popularity
    else:
        members = set(cluster_labels.loc[cluster_labels["cluster"] == user_cluster.iloc[0], "user_id"])
        segment = ratings[ratings["user_id"].isin(members)].groupby("product_id").agg(mean_rating=("rating", "mean"), rating_count=("rating", "size"))
        candidates = segment[segment["rating_count"] >= 3].sort_values(["mean_rating", "rating_count"], ascending=False)
    return candidates.loc[~candidates.index.isin(seen)].head(n).reset_index()
