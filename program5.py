
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("KNN Classification")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)
print("Predicted flower:", iris.target_names[prediction[0]])

# GRAPH 1: Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=iris.target_names
)
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.show()

# GRAPH 2: Actual flower classes
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset - Actual Classes")
plt.tight_layout()
plt.show()

