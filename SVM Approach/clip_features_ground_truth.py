import csv

# Paths
features_path = "clip_features.csv"
labels_path = "combined_fraud_nonfraud.csv"
output_path = "clip_features_with_labels.csv"

# Step 1: Load ground truth labels
id_to_label = {}
with open(labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        clean_id = row["id"].strip().replace(".jpg", "").replace(".png", "")
        id_to_label[clean_id] = row["isfraud"].strip()

# Step 2: Merge features with labels
with open(features_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "w", encoding="utf-8", newline='') as f_out:

    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames + ["isfraud"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    count = 0
    for row in reader:
        image_id = row["id"].strip()
        if image_id in id_to_label:
            row["isfraud"] = id_to_label[image_id]
            writer.writerow(row)
            count += 1
        else:
            print(f"⚠️ ID not found in label file: {image_id}")

print(f"Merged {count} records → {output_path}")
