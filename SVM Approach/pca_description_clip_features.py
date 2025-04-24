import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load feature data
path = "clip_features_with_labels.csv"
df = pd.read_csv(path)

# Prepare data
X = df.drop(columns=["id", "isfraud"]).values.astype(np.float32)
y = df["isfraud"].astype(int).values

# Step 1: PCA to 50D (to speed up t-SNE)
print("Running PCA")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)

# Step 2: t-SNE to 2D
print("Running t-SNE")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Step 3: Plot
plt.figure(figsize=(10, 6))
colors = ['blue' if label == 0 else 'red' for label in y]
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=30, label="fraud=red, non-fraud=blue")
plt.title("t-SNE Visualization of CLIP Features (PCA 50 → t-SNE 2D)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_clip_features.png")
