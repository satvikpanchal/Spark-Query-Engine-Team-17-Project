import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np
import torch

plt.style.use('default')

# Accuracy and Loss Plot
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Training Loss', color='tomato', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='dodgerblue', marker='s')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], label='Training Accuracy', color='limegreen', marker='o')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='orange', marker='s')
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history)

all_labels, all_preds, all_probs = [], [], []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().reshape(-1, 1).to(device)

        outputs = model(images)
        preds = (outputs > 0.5).float()
        probs = outputs.cpu().numpy()

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

all_preds_bin = np.round(all_preds)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds_bin)
labels = ["Non-Fraud", "Fraud"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 16, "color": "black"})

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
