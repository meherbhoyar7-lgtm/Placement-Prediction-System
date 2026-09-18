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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def find_optimal_k(X):
    wcss = []
    for k in range(1, 11):
        model = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )
        model.fit(X)
        wcss.append(model.inertia_)

    print("\nWCSS Values:")
    for k, value in zip(range(1, 11), wcss):
        print("K =", k, " WCSS = ", value)

    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, 11),
        wcss,
        marker='o'
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for optimal K")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

    return wcss


def create_model(k):
    model = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    return model


def train_model(model, X):
    labels = model.fit_predict(X)
    print("\nK-means executed successfully")
    return model, labels


def evaluate_model(model, X, labels):
    print("\nInertia (WCSS):")
    print(model.inertia_)
    print("\nIteration to Converge")
    print(model.n_iter_)

    silhouette = silhouette_score(X, labels)
    print("\nSilhouette Score:")
    print(silhouette)

    return silhouette


def display_clusters(X, labels, model):
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap='viridis',
        s=30,
        label='Data Points'
    )

    # Plot centroids
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker='X',
        s=200,
        c='red',
        label='Centroids'
    )

    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")  # Changed from X.columns to avoid numpy array error
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


def main():
    df = load_data()
    print("\nOriginal Dataset Shape:")
    print(df.shape)

    X = split_X_data(
        df,
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nProcessed Dataset Shape:")
    print(X.shape)

    numerical_features, categorical_features = identify_features(X)

    print("\nNumerical Features :")
    print(numerical_features)
    print("\nCategorical Features :")
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

    X, _, imputer = handle_missing_values(
        X,
        X.copy(),
        numerical_features
    )

    print("\nMissing Value Handling Completed.")

    X, _, scaler = standardize_data(
        X,
        X.copy(),
        numerical_features
    )
    print("\nStandardization Completed.")

    X, _, one_hot_encoder = one_hot_encode_data(
        X,
        X.copy(),
        one_hot_features
    )
    print("\nOne-Hot Encoding Completed.")

    X, _, ordinal_encoder = ordinal_encode_data(
        X,
        X.copy(),
        ordinal_features
    )
    print("\nOrdinal Encoding Completed.")

    print("\nFinding optimal K...")
    wcss = find_optimal_k(X)


if __name__ == "__main__":
    main()