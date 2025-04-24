import csv

label_path = "idlabel.csv"
meta_path = "idmeta.csv"
combined_path = "combined_fraud_nonfraud.csv"

combined_rows = []

# Load fraud entries (already have isfraud=1)
with open(label_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["isfraud"] = "1"
        combined_rows.append(row)

# Load non-fraud entries (add isfraud=0 manually)
with open(meta_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["isfraud"] = "0"
        combined_rows.append(row)

# Find all columns
all_fields = set()
for row in combined_rows:
    all_fields.update(row.keys())

all_fields = list(all_fields)

# Save to combined CSV
with open(combined_path, "w", encoding="utf-8", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(combined_rows)

print(f"Combined CSV written to: {combined_path}")
print(f"Total rows: {len(combined_rows)}")