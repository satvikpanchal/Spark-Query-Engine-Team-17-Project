import csv
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Load model and scaler
model = joblib.load("svm_clip_model.pkl")
scaler = joblib.load("svm_clip_scaler.pkl")

# Load feature data with labels
df = pd.read_csv("clip_features_with_labels.csv")
X = df.drop(columns=["id", "isfraud"]).astype(np.float32).values
y_true = df["isfraud"].astype(np.int32).values
ids = df["id"].values

# Normalize
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# Print results
print(f"\n Overall Accuracy: {(y_pred == y_true).sum()}/{len(y_true)} = {(y_pred == y_true).mean()*100:.2f}%\n")

for i in range(len(y_pred)):
    status = "Good" if y_pred[i] == y_true[i] else "Not good"
    print(f"{status} {ids[i]:<30} → Pred: {y_pred[i]} ({y_prob[i]*100:.2f}%), Truth: {y_true[i]}")

# Classification report
print(classification_report(y_true, y_pred, target_names=["non-fraud", "fraud"]))
