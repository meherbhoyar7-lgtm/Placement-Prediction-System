import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
import os

# Add the project root to the python path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.load_data import load_data

def create_model(eps=0.5, min_samples=5):
    """Create the DBSCAN model."""
    return DBSCAN(eps=eps, min_samples=min_samples)

def train_model(model, X):
    """Train the model using the fit method."""
    model.fit(X)
    return model

def evaluate_model(X, labels):
    """Evaluate the model with silhouette score."""
    # Silhouette score requires at least 2 clusters (excluding noise sometimes, but we compute on all)
    unique_labels = set(labels)
    if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
        # If there's at least 2 valid clusters, or 1 cluster + noise (2 total but score might be ill-defined, usually works if at least 2 labels exist)
        try:
            score = silhouette_score(X, labels)
            return score
        except ValueError:
            return None
    else:
        return None

def display_results(labels, score):
    """Display silhouette score and counts per cluster."""
    if score is not None:
        print(f"Silhouette Score: {score:.4f}")
    else:
        print("Silhouette Score: N/A (Requires >1 cluster)")
        
    print("\nNumber of points in each cluster (-1 represents noise/outliers):")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} points")

def main():
    # 1. Load the data
    df = load_data()
    
    # 2. Select 4 features: CGPA, PlacementStatus, and 2 more
    features = ['CGPA', 'PlacementStatus', 'AptitudeTestScore', 'CodingTestScore']
    
    # Drop rows with NaN in the selected features
    df = df.dropna(subset=features)
    
    # 3. Sample 500 rows
    df_sample = df.sample(n=500, random_state=42)
    
    X_raw = df_sample[features]
    
    # 4. Preprocessing: Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 5. Call model
    # Using eps=1.0 and min_samples=5 as starting parameters
    model = create_model(eps=1.0, min_samples=5)
    
    # 6. Training part
    model = train_model(model, X_scaled)
    
    # 7. Evaluate
    score = evaluate_model(X_scaled, model.labels_)
    
    # 8. Show results (with regard to each cluster how many points are there)
    display_results(model.labels_, score)

if __name__ == "__main__":
    main()