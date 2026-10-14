
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11, 13, 15, 17])

# Create models
models = {
    "Linear Regression": make_pipeline(
        PolynomialFeatures(degree=5),
        LinearRegression()
    ),
    "Ridge Regression": make_pipeline(
        PolynomialFeatures(degree=5),
        Ridge(alpha=10)
    ),
    "Lasso Regression": make_pipeline(
        PolynomialFeatures(degree=5),
        Lasso(alpha=0.1, max_iter=100000)
    )
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(name)
    print("Mean Squared Error:", round(mse, 3))
    print()

# Smooth values for plotting
X_plot = np.linspace(1, 8, 200).reshape(-1, 1)

plt.scatter(X, y, color="black", label="Actual Data")

for name, model in models.items():
    plt.plot(X_plot, model.predict(X_plot), label=name)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regularization: Ridge and Lasso")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
