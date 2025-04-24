import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load CLIP feature data
df = pd.read_csv("clip_features_with_labels.csv")
X = df.drop(columns=["id", "isfraud"]).astype(np.float32).values
y = df["isfraud"].astype(np.int32).values

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Train SVM
clf = SVC(kernel="rbf", C=1.0, probability=True)
print("Training SVM on CLIP features")
clf.fit(X_train, y_train)

# Evaluation
def evaluate(name, model, X, y):
    print(f"\n{name} Set Evaluation")
    y_pred = model.predict(X)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred, target_names=["non-fraud", "fraud"]))

evaluate("Train", clf, X_train, y_train)
evaluate("Val", clf, X_val, y_val)
evaluate("Test", clf, X_test, y_test)

# Save model and scaler
joblib.dump(clf, "svm_clip_model.pkl")
joblib.dump(scaler, "svm_clip_scaler.pkl")
print("Saved SVM model and scaler!")
