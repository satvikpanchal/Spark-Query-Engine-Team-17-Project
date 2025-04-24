import os
import csv
import base64
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import sys
csv.field_size_limit(sys.maxsize)

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Convert base64 string to CLIP embedding
def base64_to_clip_vector(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
        return feats.squeeze().cpu().numpy()
    except Exception as e:
        print("Error decoding image:", e)
        return None

# Input / Output Paths
input_csv = "idimage.csv"
output_csv = "clip_features.csv"

# Write features to CSV
with open(input_csv, "r", encoding="utf-8") as infile, \
     open(output_csv, "w", encoding="utf-8", newline='') as outfile:

    reader = csv.DictReader(infile)
    fieldnames = ["id"] + [f"feat_{i}" for i in range(512)]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    count = 0
    for row in tqdm(reader, desc="Extracting CLIP features"):
        img_id = row["name"].replace(".jpg", "").replace(".png", "")
        base64_str = row["imageData"]

        vec = base64_to_clip_vector(base64_str)
        if vec is not None:
            row_data = {"id": img_id}
            row_data.update({f"feat_{i}": vec[i] for i in range(512)})
            writer.writerow(row_data)
            count += 1

print(f"Saved CLIP features for {count} images → {output_csv}")
