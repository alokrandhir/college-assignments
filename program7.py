
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

X = df[["Study_Hours"]]
y = df["Marks"]

# Scale the data
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()

# Train SVM regression model
model = SVR(kernel="rbf", C=100, epsilon=0.05)
model.fit(X_scaled, y_scaled)

# Predict
y_pred_scaled = model.predict(X_scaled)
y_pred = y_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()

print("SVM Regression")
print("Mean Squared Error:", round(mean_squared_error(y, y_pred), 2))
print("R2 Score:", round(r2_score(y, y_pred), 3))

# Predict marks for 9 study hours
new_hours = pd.DataFrame({"Study_Hours": [9]})
new_hours_scaled = X_scaler.transform(new_hours)
prediction_scaled = model.predict(new_hours_scaled)
prediction = y_scaler.inverse_transform(
    prediction_scaled.reshape(-1, 1)
)[0, 0]

print("Predicted marks for 9 hours:", round(prediction, 2))

# Plot results
X_plot = pd.DataFrame({
    "Study_Hours": [
        i / 10 for i in range(10, 91)
    ]
})
X_plot_scaled = X_scaler.transform(X_plot)

y_plot_scaled = model.predict(X_plot_scaled)
y_plot = y_scaler.inverse_transform(
    y_plot_scaled.reshape(-1, 1)
).ravel()

plt.scatter(X["Study_Hours"], y, color="blue", label="Actual Marks")
plt.plot(X_plot["Study_Hours"], y_plot, color="red", label="SVR Curve")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("SVM Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
