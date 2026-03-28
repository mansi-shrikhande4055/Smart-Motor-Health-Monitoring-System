# =========================
# ADVANCED MOTOR HEALTH SYSTEM
# =========================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# Load dataset
# -------------------------
data = pd.read_csv("motor_data.csv")
data = data.dropna()

# -------------------------
# Visualization
# -------------------------
plt.scatter(data['vibration'], data['temperature'], c=data['fault'])
plt.xlabel("Vibration")
plt.ylabel("Temperature")
plt.title("Motor Health Visualization")
plt.show()

# -------------------------
# Features & Labels
# -------------------------
X = data[['vibration', 'temperature', 'current']]
y = data['fault']

# -------------------------
# Scaling (NEW 🔥)
# -------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Model 1: Random Forest
# -------------------------
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# -------------------------
# Model 2: Logistic Regression
# -------------------------
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

# -------------------------
# Results Comparison 🔥
# -------------------------
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

# -------------------------
# Confusion Matrix
# -------------------------
print("\nConfusion Matrix (RF):")
print(confusion_matrix(y_test, rf_pred))

# -------------------------
# Classification Report
# -------------------------
print("\nClassification Report (RF):")
print(classification_report(y_test, rf_pred))

# -------------------------
# Feature Importance
# -------------------------
importance = rf_model.feature_importances_
features = ['vibration', 'temperature', 'current']

print("\nFeature Importance:")
for i in range(len(features)):
    print(features[i], ":", importance[i])

# -------------------------
# Smart Prediction Input
# -------------------------
sample = pd.DataFrame([[0.8, 50, 14]],
                      columns=['vibration', 'temperature', 'current'])

sample_scaled = scaler.transform(sample)

result = rf_model.predict(sample_scaled)

print("\nSample Prediction (0=Healthy, 1=Fault):", result)
# -------------------------
# Fault Interpretation (NEW 🔥)
# -------------------------

v, t, c = sample.values[0]

print("\n--- Fault Analysis ---")

if result[0] == 0:
    print("Motor is Healthy ✅")
else:
    print("Motor Fault Detected ⚠️")
    
    if v > 1.5:
        print("Possible Cause: Bearing Fault (High Vibration)")
    if t > 65:
        print("Possible Cause: Overheating / Stator Fault")
    if c > 20:
        print("Possible Cause: Overload Condition")