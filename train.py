"""Train clustering and recommendation artifacts from a ratings CSV.

Example:
  python train.py --data C:\\Users\\Yaswanth\\Downloads\\ratings.csv --max-rows 1000000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from src.data import load_sample, user_features
from src.models import build_item_knn, compare_clusterers


def main() -> None:
    parser = argparse.ArgumentParser(description="Train product recommendation project artifacts.")
    parser.add_argument("--data", required=True, help="Path to ratings.csv")
    parser.add_argument("--out", default="artifacts", help="Output artifact directory")
    parser.add_argument("--max-rows", type=int, default=1_000_000, help="Maximum valid ratings used for model training")
    parser.add_argument("--clusters", type=int, default=5, help="Number of user segments")
    parser.add_argument("--min-product-ratings", type=int, default=5)
    parser.add_argument("--min-user-ratings", type=int, default=3)
    args = parser.parse_args()
    if args.max_rows < 100 or args.clusters < 2:
        parser.error("--max-rows must be >= 100 and --clusters must be >= 2")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ratings = load_sample(args.data, args.max_rows)
    features = user_features(ratings)
    if len(features) <= args.clusters:
        raise ValueError("Not enough distinct users for the requested number of clusters.")

    metrics, results, _ = compare_clusterers(features, args.clusters)
    winner = metrics.iloc[0]["model"]
    labels = results[winner].labels
    cluster_labels = features[["user_id"]].assign(cluster=labels)
    features.assign(cluster=labels).to_csv(out / "user_features.csv", index=False)
    cluster_labels.to_csv(out / "user_clusters.csv", index=False)
    metrics.to_csv(out / "clustering_metrics.csv", index=False)

    knn, matrix, products, popularity = build_item_knn(ratings, args.min_product_ratings, args.min_user_ratings)
    joblib.dump(knn, out / "item_knn.joblib")
    save_npz(out / "product_user_matrix.npz", matrix)
    np.save(out / "products.npy", products.to_numpy())
    popularity.to_csv(out / "popularity.csv")
    ratings.to_csv(out / "training_ratings.csv", index=False)
    metadata = {
        "rows_used": int(len(ratings)), "users": int(ratings.user_id.nunique()), "products": int(ratings.product_id.nunique()),
        "best_clustering_model": winner, "n_clusters": args.clusters,
        "rating_scale": [float(ratings.rating.min()), float(ratings.rating.max())],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
