import numpy as np
import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess import (
    split_X_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def create_model(k):
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )
    return model

def train_model(model, X):
    model.fit(X)
    print("\nAgglomerative clustering complete.")
    return model

def evaluate_model(model, X):
    labels = model.labels_
    score = silhouette_score(X, labels)

    print("Silhouette score:")
    print(score)
    return labels

def display_dendrogram(X, cut_distance=None):
    linked = linkage(X, method="ward")
    plt.figure(figsize=(12, 6))
    if cut_distance:
        dendrogram(linked, truncate_mode="level", p=5, color_threshold=cut_distance)
        plt.axhline(y=cut_distance, color='r', linestyle='--')
    else:
        dendrogram(linked, truncate_mode="level", p=5)
    plt.title("Dendrogram")
    plt.show()

def main():
    df = load_data()
    print("Dataset Shape:")
    print(df.shape)

    # Sample data to avoid memory issues with hierarchical clustering
    df = df.sample(n=500, random_state=42)

    X = split_X_data(
        df,
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    numerical_features, categorical_features = identify_features(X)
    print("\nNumerical features:")
    print(numerical_features)
    print("\nCategorical features:")
    print(categorical_features)

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]
    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    X, _, imputer = handle_missing_values(X, X.copy(), numerical_features)
    X, _, scaler = standardize_data(X, X.copy(), numerical_features)
    X, _, one_hot_encoder = one_hot_encode_data(X, X.copy(), one_hot_features)
    X, _, ordinal_encoder = ordinal_encode_data(X, X.copy(), ordinal_features)
    
    print("\nData preprocessing complete.")
    
    print("\nDisplaying dendrogram...")
    display_dendrogram(X, cut_distance=30)
    
    k = 3
    model = create_model(k)
    model = train_model(model, X)
    labels = evaluate_model(model, X)

if __name__ == "__main__":
    main()