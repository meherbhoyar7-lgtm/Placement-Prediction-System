from src.data.load import load_data
from src.data.preprocess import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def create_model():
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )
    return model

def train_model(model, X):
    model.fit(X)
    print("\nAgglomerative clustering complete.:")
    return model

def evaluate_model(model, X):
    labels = model.labels_
    score = silhouette_score(
        X,
        labels,
    )

    print("Silhouette score:")
    print(score)
    return labels

def display_dendrogram(X, cut_distance):
    linked = linkage(
        X,
        method="ward"
    )
    plt.figure(figsize=(12, 6))
    dendrogram(
        linked,
        truncate_mode="level",
        lin
    )

def main():
    df = load_data()
    print("Dataset Shape:")
    print(df.shape)

    df = df.sample(
        n=500,
        random_state=42,
    )

    X = split_X_data(
        df,
        drop_columns=(
            "StudentId",
            "PlacementStatus",
            "",
            ""
        )
    )

    numerical_features, categorical_features = (
        identify_features(X)
    )
    print("Numerical features:")
    print(numerical_features)
    print("Categorical features:")
    print(categorical_features)

    one_hot_features=[

    ]