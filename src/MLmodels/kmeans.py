import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 1. Dynamically resolve the path to the data folder
# This goes up two directories from kmeans.py (MLmodels -> src -> PlacementpredictionSystem)
# and then into the data folder.
current_dir = Path(__file__).parent
file_path = current_dir.parent.parent / 'data' / 'placement_data.csv'

df = pd.read_csv(file_path)

features = [
    "CGPA",
    "AttendancePercent", # <-- Fixed spelling here
    "Projects",
    "CodingTestScore"
]
X = df[features].dropna()

print("Selected Features:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

X['cluster'] = clusters

# 2. Fixed the newline escape sequence
print("\nCluster Assignment:")
print(X.head(10))

# 3. Fixed the undefined variable issue
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(
    centers,
    columns=features
)

print("\nCluster Centers:")
print(centers_df)
print("/Studets in Eacg cluster:")
print(X["cluster"].value_counts().sort_index())
plt.figure(figsize=(3, 6))

plt.scatter(
    X["CGPA"],
    X["CodingTestScore"], # (or whichever feature you are plotting on the Y-axis)
    c=X["cluster"],       # <--- Make sure this starts with c=
    cmap="viridis",
    s=50
)

plt.xlabel("CGPA")
plt.ylabel("Coding Test Score")
plt.title("K Means Clustering or Students")

plt.show()