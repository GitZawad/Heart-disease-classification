import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

df = pd.read_csv(
    "processed.cleveland.data",
    header=None,
    names=columns
)

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Show missing values
print("Missing values in each column:\n")
print(df.isnull().sum())

# Convert columns to numeric
df["ca"] = pd.to_numeric(df["ca"])
df["thal"] = pd.to_numeric(df["thal"])

# Remove rows with missing values
df.dropna(inplace=True)

# Check the cleaned dataset
print("\nDataset shape after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Convert target to binary
df["target"] = (df["target"] > 0).astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts())

# Figure 1: Heart Disease Distribution

plt.figure(figsize=(6,4))

df["target"].value_counts().plot(kind="bar")

plt.title("Heart Disease Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Patients")

plt.xticks(
    [0,1],
    ["No Disease", "Heart Disease"],
    rotation=0
)

plt.tight_layout()

plt.savefig("heart_disease_distribution.png")

plt.show()

# Figure 2: Age Distribution

plt.figure(figsize=(8,5))

plt.hist(df["age"], bins=10)

plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("age_distribution.png")

plt.show()

# Figure 3: Correlation Heatmap

plt.figure(figsize=(12, 8))

corr = df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Heart Disease Dataset")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()

# Separate features and target

X = df.drop("target", axis=1)
y = df["target"]

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Standardize features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining set shape:")
print(X_train.shape)

print("\nTesting set shape:")
print(X_test.shape)

# ===============================
# Logistic Regression
# ===============================

lr = LogisticRegression(max_iter=1000)

# Train the model
lr.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test_scaled)

# Evaluate the model
print("\n===== LOGISTIC REGRESSION =====")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred_lr))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Figure 4: Logistic Regression Confusion Matrix

cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title("Logistic Regression Confusion Matrix")

plt.colorbar()

plt.xticks(
    [0,1],
    ["No Disease", "Disease"]
)

plt.yticks(
    [0,1],
    ["No Disease", "Disease"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i,j],
            ha='center',
            va='center'
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_lr.png")

plt.show()

# ===============================
# Random Forest
# ===============================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# Results
print("\n===== RANDOM FOREST =====")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_rf
))

# Figure 5: Random Forest Confusion Matrix

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title("Random Forest Confusion Matrix")

plt.colorbar()

plt.xticks(
    [0,1],
    ["No Disease", "Disease"]
)

plt.yticks(
    [0,1],
    ["No Disease", "Disease"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i,j],
            ha='center',
            va='center'
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_rf.png")

plt.show()

# Figure 6: Accuracy Comparison

models = [
    "Logistic Regression",
    "Random Forest"
]

accuracies = [
    0.8667,
    0.8833
]

plt.figure(figsize=(7,5))

plt.bar(models, accuracies)

plt.ylabel("Accuracy")
plt.title("Comparison of Classification Accuracy")

plt.ylim(0.80, 0.90)

plt.tight_layout()

plt.savefig("accuracy_comparison.png")

plt.show()