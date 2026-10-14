
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create dataset
data = {
    "Height": [150, 160, 165, 170, 175, 180],
    "Weight": [50, 55, 60, 65, 70, 75],
    "Age": [20, 21, 22, 23, 24, 25]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA and reduce to 2 components
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Display reduced dataset
result = pd.DataFrame(
    reduced_data,
    columns=["PC1", "PC2"]
)

print("\nReduced Dataset:")
print(result)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Retained:")
print(round(sum(pca.explained_variance_ratio_), 3))

# Visualize the components
plt.scatter(result["PC1"], result["PC2"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Dimensionality Reduction")
plt.grid(True)
plt.show()
