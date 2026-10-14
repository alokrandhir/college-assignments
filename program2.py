
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
data = {
    "Income": [15, 18, 20, 25, 70, 75, 80, 85],
    "Spending": [20, 25, 18, 30, 75, 80, 70, 85]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[["Income", "Spending"]])

print("\nClustered Dataset:")
print(df)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualize clusters
plt.scatter(
    df["Income"], df["Spending"],
    c=df["Cluster"], cmap="viridis", s=100
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="red", marker="X", s=200, label="Centroids"
)

plt.xlabel("Annual Income (in thousands)")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.legend()
plt.show()
