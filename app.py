"""Streamlit deployment for clustering comparison and product recommendations."""
from __future__ import annotations
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(_file_), 'src'))
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.sparse import load_npz
from models import recommendations_for_user, similar_products

ARTIFACTS = Path("artifacts")
st.set_page_config(page_title="Product Recommendation System", page_icon="⭐", layout="wide")


@st.cache_resource
def load_artifacts():
    needed = ["metadata.json", "item_knn.joblib", "product_user_matrix.npz", "products.npy", "popularity.csv", "training_ratings.csv", "user_clusters.csv", "clustering_metrics.csv", "user_features.csv"]
    absent = [name for name in needed if not (ARTIFACTS / name).exists()]
    if absent:
        return None, absent
    return {
        "meta": json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8")),
        "knn": joblib.load(ARTIFACTS / "item_knn.joblib"), "matrix": load_npz(ARTIFACTS / "product_user_matrix.npz"),
        "products": pd.Index(np.load(ARTIFACTS / "products.npy", allow_pickle=True)),
        "popularity": pd.read_csv(ARTIFACTS / "popularity.csv"), "ratings": pd.read_csv(ARTIFACTS / "training_ratings.csv", dtype={"user_id": "string", "product_id": "string"}),
        "clusters": pd.read_csv(ARTIFACTS / "user_clusters.csv", dtype={"user_id": "string"}),
        "metrics": pd.read_csv(ARTIFACTS / "clustering_metrics.csv"), "features": pd.read_csv(ARTIFACTS / "user_features.csv", dtype={"user_id": "string"}),
    }, []


st.title("Product Recommendation System")
art, missing = load_artifacts()
if art is None:
    st.warning("Model artifacts are not available yet.")
    st.code("python train.py --data C:\\Users\\Yaswanth\\Downloads\\ratings.csv --max-rows 1000000\nstreamlit run app.py", language="bash")
    st.stop()

st.caption(f"Trained on {art['meta']['rows_used']:,} ratings | Best segmentation model: {art['meta']['best_clustering_model']}")
tab1, tab2, tab3 = st.tabs(["Similar Products", "For a User", "Clustering Analysis"])

with tab1:
    product = st.selectbox("Choose a product ID", art["products"].tolist(), index=0)
    result = similar_products(product, art["knn"], art["matrix"], art["products"], art["popularity"].set_index("product_id"), 10)
    st.dataframe(result, hide_index=True, use_container_width=True, column_config={"similarity": st.column_config.NumberColumn(format="%.3f"), "mean_rating": st.column_config.NumberColumn(format="%.2f")})

with tab2:
    user = st.selectbox("Choose a user ID", sorted(art["clusters"]["user_id"].unique()))
    result = recommendations_for_user(user, art["ratings"], art["clusters"], art["popularity"].set_index("product_id"), 10)
    st.dataframe(result, hide_index=True, use_container_width=True, column_config={"mean_rating": st.column_config.NumberColumn(format="%.2f")})

with tab3:
    left, right = st.columns(2)
    with left:
        st.subheader("Model comparison")
        st.dataframe(art["metrics"], hide_index=True, use_container_width=True)
        st.caption("Higher silhouette and Calinski-Harabasz scores are better; lower Davies-Bouldin is better.")
    with right:
        st.subheader("Selected user segments")
        plot_data = art["features"].merge(art["clusters"], on="user_id")
        fig = px.scatter(plot_data, x="rating_count", y="mean_rating", color="cluster_x", hover_data=["unique_products", "high_rating_share"], log_x=True)
        st.plotly_chart(fig, use_container_width=True)
