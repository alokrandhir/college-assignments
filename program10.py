
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample multimodal dataset
data = {
    "text": [
        "Government announces new public transport plan",
        "Scientists publish results of climate research",
        "City opens a new public hospital",
        "Education department announces school improvements",
        "Official report confirms new railway project",
        "Shocking secret miracle cure hidden from everyone",
        "Celebrity secretly becomes ruler of the moon",
        "Fake photograph proves aliens control the city",
        "Unbelievable story claims impossible events happened",
        "Viral rumour says gravity will disappear tomorrow",
        "Researchers publish verified medical study",
        "City council announces new public library",
        "Secret conspiracy claims all birds are robots",
        "Online rumour claims water turns into gold",
        "Official agency releases annual economic report",
        "A viral post claims humans can breathe underwater",
        "University announces new engineering research",
        "Government publishes updated road safety rules",
        "An anonymous post claims the sun will vanish",
        "Health department releases vaccination information"
    ],
    "image_brightness": [
        120, 135, 110, 145, 125, 230, 240, 220, 235, 245,
        130, 115, 250, 225, 140, 238, 125, 132, 248, 118
    ],
    "image_red": [
        100, 110, 90, 120, 105, 220, 230, 210, 225, 235,
        108, 95, 240, 215, 115, 228, 100, 107, 238, 98
    ],
    "label": [
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 0, 1, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[["text", "image_brightness", "image_red"]],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# Process text and image features separately
preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(), "text"),
    ("image", StandardScaler(), ["image_brightness", "image_red"])
])

# Create and train classifier
model = Pipeline([
    ("features", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("MULTIMODAL FAKE NEWS DETECTION")
print("--------------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Fake News", "Real News"],
    zero_division=0
))

# Predict a sample news item
sample = pd.DataFrame({
    "text": ["Secret rumour claims a miracle cure is being hidden"],
    "image_brightness": [235],
    "image_red": [225]
})

result = model.predict(sample)[0]
print("Sample news prediction:",
      "Real News" if result == 1 else "Fake News")

print("\nProgram completed successfully.")
