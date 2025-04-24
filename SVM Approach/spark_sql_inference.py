from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, year, to_date
from pyspark.sql.types import IntegerType, DoubleType
import joblib
import numpy as np

# Spark session
spark = SparkSession.builder \
    .appName("CLIP Fraud Detection") \
    .getOrCreate()

# Trained model and scalar
model = joblib.load("svm_clip_model.pkl")
scaler = joblib.load("svm_clip_scaler.pkl")

# Prediction UDF
def predict_label_prob(*features):
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0][1]
    label = int(prob > 0.5)
    return float(prob), label

predict_label_udf = udf(lambda *f: predict_label_prob(*f)[1], IntegerType())
predict_prob_udf = udf(lambda *f: predict_label_prob(*f)[0], DoubleType())

# Combined datasets
df_features = spark.read.option("header", True).csv("clip_features.csv", inferSchema=True)
df_metadata = spark.read.option("header", True).csv("combined_fraud_nonfraud.csv", inferSchema=True)

# Join by ID
df = df_features.join(df_metadata, on="id")

# Inference
feature_columns = [f"feat_{i}" for i in range(512)]

df = df \
    .withColumn("probability", predict_prob_udf(*[col(c) for c in feature_columns])) \
    .withColumn("label", predict_label_udf(*[col(c) for c in feature_columns]))

# Birth year new format
df = df.withColumn("birth_year", year(to_date(col("birthday"), "yyyy-MM-dd")))

# Predictions save
df.write.mode("overwrite").option("header", True).csv("output_predictions.csv")

# SQL View
df.createOrReplaceTempView("predictions")

# -----------------------------------------------------------------------------------------------------------------
# QUERIES
# 1. Total Fraud vs Non-Fraud Counts
spark.sql("""
    SELECT label, COUNT(*) AS doc_count
    FROM predictions
    GROUP BY label
    ORDER BY label DESC
""").show()

# 2. Ethnicity Fraud + Non-Fraud Breakdown
spark.sql("""
    SELECT ethnicity,
           SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS frauds,
           SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS non_frauds,
           COUNT(*) AS total_docs,
           ROUND(SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate
    FROM predictions
    WHERE ethnicity IS NOT NULL
    GROUP BY ethnicity
    ORDER BY fraud_rate DESC
""").show()

# 3. Class × Eye Color × Ethnicity Breakdown with Both Labels
spark.sql("""
    SELECT ethnicity, class, eye_color,
           COUNT(*) AS total,
           SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS frauds,
           SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS non_frauds,
           ROUND(AVG(probability), 4) AS avg_prob
    FROM predictions
    WHERE ethnicity IS NOT NULL AND class IS NOT NULL AND eye_color IS NOT NULL
    GROUP BY ethnicity, class, eye_color
    ORDER BY total DESC
    LIMIT 10
""").show()

# 4. Class-Level Fraud Rate with Total/Label Split
spark.sql("""
    SELECT class,
           COUNT(*) AS total,
           SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS frauds,
           SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS non_frauds,
           ROUND(SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS fraud_rate
    FROM predictions
    WHERE class IS NOT NULL
    GROUP BY class
    ORDER BY fraud_rate DESC
""").show()
