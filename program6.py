
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = df[["Study_Hours"]]
y = df["Marks"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict marks
y_pred = model.predict(X)

# Display results
print("Linear Regression")
print("Slope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))
print("R2 Score:", round(r2_score(y, y_pred), 3))
print("Mean Squared Error:", round(mean_squared_error(y, y_pred), 2))

# Predict for 9 study hours
prediction = model.predict(pd.DataFrame({"Study_Hours": [9]}))
print("Predicted marks for 9 hours:", round(prediction[0], 2))

# Plot actual data and regression line
plt.scatter(X["Study_Hours"], y, color="blue", label="Actual Marks")
plt.plot(X["Study_Hours"], y_pred, color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
