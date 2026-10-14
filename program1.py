
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

# Display dataset
print("Dataset:")
print(df)

# Calculate correlation
correlation, p_value = pearsonr(
    df["Study_Hours"], df["Marks"]
)

print("\nCorrelation:", round(correlation, 3))
print("P-value:", round(p_value, 4))

# Plot the relationship
plt.scatter(df["Study_Hours"], df["Marks"])
plt.xlabel("Study Hours (Independent Variable)")
plt.ylabel("Marks (Dependent Variable)")
plt.title("Study Hours vs Marks")
plt.grid(True)
plt.show()
