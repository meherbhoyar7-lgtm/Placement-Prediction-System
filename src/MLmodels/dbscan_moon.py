from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt


def create_data():
    X, _ = make_moons(
        n_samples=300,
        noise=0.06,
        random_state=42
    )
    return X

def create_kmeans_model():
    kmeans = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=42
    )
    return kmeans

def create_dbscan_model():
    dbscan = DBSCAN(
        min_samples=5,
        metric='euclidean',
        eps=0.15
    )
    return dbscan

def plot_results(X, kmeans_labels, dbscan_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=30)
    ax1.set_title("KMeans Clustering")
    
    ax2.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=30)
    ax2.set_title("DBSCAN Clustering")
    
    plt.tight_layout()
    plt.show()

def main():
    X = create_data()
    
    kmeans = create_kmeans_model()
    kmeans_labels = kmeans.fit_predict(X)
    
    dbscan = create_dbscan_model()
    dbscan_labels = dbscan.fit_predict(X)
    
    plot_results(X, kmeans_labels, dbscan_labels)

if __name__ == "__main__":
    main()