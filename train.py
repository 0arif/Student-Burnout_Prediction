import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("dataset/burnout_dataset.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== DATASET INFO ==========\n")
df.info()

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

print(df["Burnout_Level"].value_counts())

# =====================================
# Visualization
# =====================================

plt.figure(figsize=(6,4))

df["Burnout_Level"].value_counts().plot(kind="bar")

plt.title("Burnout Level Distribution")
plt.xlabel("Burnout Level")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()

numeric_features = [
    "Study_Hours",
    "Sleep_Hours",
    "Attendance",
    "Assignment_Load",
    "Social_Media_Hours",
    "Exercise_Hours",
    "Stress_Level"
]

df[numeric_features].hist(figsize=(12,8))

plt.tight_layout()
plt.show()

# =====================================
# Label Encoding
# =====================================

df["Burnout_Level"] = df["Burnout_Level"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

# =====================================
# Correlation
# =====================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")

plt.tight_layout()
plt.show()

# =====================================
# Feature & Target
# =====================================

X = df.drop("Burnout_Level", axis=1)

y = df["Burnout_Level"]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Random Forest Model
# =====================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# Prediction
# =====================================

y_pred = model.predict(X_test)

# =====================================
# Accuracy
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========\n")
print(f"Accuracy : {accuracy * 100:.2f}%")

# =====================================
# Confusion Matrix
# =====================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low","Medium","High"],
    yticklabels=["Low","Medium","High"]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()

# =====================================
# Classification Report
# =====================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low","Medium","High"]
    )
)

# =====================================
# Save Model
# =====================================

model_data = {
    "model": model,
    "feature_names": X.columns.tolist()
}

joblib.dump(model_data, "models/burnout_model.pkl")

print("\nModel saved successfully!")

print("\nLocation : models/burnout_model.pkl")