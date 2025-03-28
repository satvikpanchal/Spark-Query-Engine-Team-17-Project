from pyspark.sql import SparkSession
import os
import logging
import time  # ⏱️ for timing

# Silence Spark logs
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

import pyspark
pyspark.SparkContext.setSystemProperty('spark.ui.showConsoleProgress', 'false')

# Create Spark session
spark = SparkSession.builder \
    .appName("IDNet Fraud Detection") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load the CSV file
df_label = spark.read.option("header", "true").csv("idlabel.csv")
df_label.printSchema()
df_label.show(5)

# ⏱️ Start timer before query logic
start_time = time.time()

# Fraudulent entries
df_fraud = df_label.filter(df_label["fraudpattern"].isNotNull())
fraudulent_count = df_fraud.count()

# Fraud counts by type
pattern_counts = df_fraud.groupBy("fraudpattern").count()

# ⏱️ End timer after queries
end_time = time.time()
elapsed_time = end_time - start_time

# Output
print("\n=== Summary ===")
print(f"Fraudulent documents: {fraudulent_count}")
print("\nFraud pattern counts:")
pattern_counts.show(truncate=False)

print(f"\n⏱️ Query execution time: {elapsed_time:.2f} seconds")
