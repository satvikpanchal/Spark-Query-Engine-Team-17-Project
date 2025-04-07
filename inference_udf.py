from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, regexp_replace, when, length
from pyspark.sql.types import FloatType
from spark_udfs.inference_udf import InferenceUDF
from helper_functions import inference_wrapper
import logging
import sys



logging.basicConfig(filename="fraud_inference.log", level=logging.INFO, filemode="w")
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("FraudInference")


spark = SparkSession.builder \
    .appName("FraudInference") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .getOrCreate()


# 2. Load CSVs
# df_image = spark.read.option("header", True).csv("idimage.csv")
logger.info("=== loading data ... ===")
sys.stdout.flush()
df_image = spark.read.parquet("idimage.parquet")
# df_image = spark.read.parquet("idimage.parquet").select("name", "imageData").limit(100)

df_label = spark.read.option("header", True).csv("idlabel.csv")
df_meta = spark.read.option("header", True).csv("idmeta.csv")

null_count = df_image.filter(col("name").isNull()).count()
logger.info("Null id count in df_image before preprocessing: %d", null_count)

null_count = df_label.filter(col("id").isNull()).count()
logger.info("Null id count in df_label before preprocessing: %d", null_count)

null_count = df_meta.filter(col("id").isNull()).count()
logger.info("Null id count in df_meta before preprocessing: %d", null_count)

logger.info("original df_image rows: %d", df_image.count())
logger.info("original df_label rows: %d", df_label.count())
logger.info("original df_meta rows: %d", df_meta.count())


# 3. Preprocess and combine metadata
logger.info("=== pre-processing and combining data ... ===")
df_image = df_image.withColumnRenamed("name", "id")
df_image = df_image.withColumn("id", regexp_replace("id", "\\.jpg|\\.png", ""))

df_label = df_label.withColumn("label", col("isfraud").cast("boolean").cast("int")).dropna(subset=["id"])
df_meta = df_meta.withColumn("label", lit(0)).withColumn("fraudpattern", lit(None)).dropna(subset=["id"])

df_combined_meta = df_label.select("id", "label", "fraudpattern").unionByName(
    df_meta.selectExpr("id", "label", "null as fraudpattern"))
df_combined_meta.groupBy("label").count().show()

df_final = df_image.join(df_combined_meta, on="id", how="left").fillna({"label": 0})

logger.info("df_image rows: %d", df_image.count())
logger.info("df_label rows: %d", df_label.count())
logger.info("df_meta rows: %d", df_meta.count())
logger.info("df_combined_meta rows: %d", df_combined_meta.count())
logger.info("Rows after join (df_final): %d", df_final.count())


# 4. Check nulls in loaded data
logger.info("=== Checking preprocessed data ===")
df_image.printSchema()
logger.info("df_image schema printed")
null_count = df_image.filter(col("imageData").isNull()).count()
logger.info("Null imageData count in df_image: %d", null_count)

# Check nulls in label CSV
df_label.printSchema()
logger.info("df_label schema printed")
df_label.select("id", "isfraud").show(5, truncate=False)
logger.info("Top 5 df_label with ids printed")
null_count = df_label.filter(col("isfraud").isNull()).count()
logger.info("Null isfraud count in df_label: %d", null_count)
null_count = df_label.filter(col("label").isNull()).count()
logger.info("Null label count in df_label: %d", null_count)


# Check nulls in meta CSV
df_meta.printSchema()
logger.info("df_meta schema printed")
df_meta.select("id").show(5, truncate=False)
logger.info("Top 5 df_meta with ids printed")
null_count = df_meta.filter(col("label").isNull()).count()
logger.info("Null label count in df_meta: %d", null_count)


# Check nulls in combined meta CSV
df_combined_meta.printSchema()
logger.info("df_combined_meta schema printed")
df_combined_meta.select("id", "label").show(5, truncate=False)
logger.info("Top 5 df_combined with labels and ids printed")
null_count = df_combined_meta.filter(col("id").isNull()).count()
logger.info("Null id count in df_combined_meta: %d", null_count)
null_count = df_combined_meta.filter(col("label").isNull()).count()
logger.info("Null label count in df_combined_meta: %d", null_count)


# Check nulls in final merged CSV
df_final.printSchema()
logger.info("df_final schema printed")
null_count = df_final.filter(col("imageData").isNull()).count()
logger.info("Null imageData count in df_final: %d", null_count)
# df_final.filter(col("imageData").isNull()).select("id").show(20, truncate=False)
# logger.info("top 20 null image data of df_final printed")




# 5. Define and use UDF
# Sample inference
sample_ids = df_final.select("id").sample(0.01).limit(5)
sample_df = df_final.join(sample_ids, on="id")
logger.info("Running inference on sample rows: %d", sample_df.count())

skipped = df_final.filter(col("imageData").isNull()).count()
logger.info(f"Skipped {skipped} rows due to null imageData")


# inference = InferenceUDF("fraud_model_weights.pth")
# inference_spark_udf = udf(lambda b64, image_id: inference(b64, image_id), FloatType())

logger.info("Loading inference model and preparing UDF.")
inference = InferenceUDF("fraud_model_weights.pth")

inference_spark_udf = udf(inference_wrapper, FloatType())

df_sample_pred = sample_df.withColumn("prediction", inference_spark_udf(col("imageData"), col("id")))
logger.info("Sample prediction rows: %d", df_sample_pred.count())


# Complete inference
# inference = InferenceUDF("fraud_model_weights.pth")
# inference_spark_udf = udf(lambda b64: inference(b64), FloatType())
# df_with_pred = df_final.withColumn("prediction", inference_spark_udf(col("imageData")))



# 6. Show and query results
# df_with_pred.select("id", "label", "fraudpattern", "prediction").show(10)
# df_with_pred.groupBy("fraudpattern").avg("prediction").show()
# df_with_pred.filter("prediction > 0.9").show()


df_sample_pred = df_sample_pred.drop("imageData")
logger.info("imageData dropped in Sample prediction data")

# Just select the light parts before triggering any action
df_light = df_sample_pred.select("label", "prediction")

# # Force it to be computed and written as a new checkpointed dataframe
# df_light.write.mode("overwrite").parquet("intermediate/light_predictions.parquet")
# logger.info("Saved light DataFrame to Parquet")
#
# # Reload to cut lineage
# df_light_fresh = spark.read.parquet("intermediate/light_predictions.parquet")
# logger.info("Reloaded fresh light DataFrame")
#
# # Now safely collect
# df_light_pd = df_light_fresh.limit(5).toPandas()
# print(df_light_pd)

# df_sample_pred.groupBy("label").avg("prediction").show()
# df_light = df_sample_pred.select("label", "prediction")

df_avg_pred = df_light.groupBy("label").avg("prediction")
df_avg_pred.write.mode("overwrite").csv("outputs/avg_prediction_per_label.csv", header=True)
logger.info("Saved average prediction per label.")

df_high_pred = df_light.filter(col("prediction") > 0.9)
df_high_pred.write.mode("overwrite").csv("outputs/predictions_gt_0.9.csv", header=True)
logger.info("Saved high prediction rows (prediction > 0.9).")


# 7. Save output
# df_sample_pred.write.mode("overwrite").csv("sample_output_predictions")
# logger.info("Sample predictions saved to sample_output_predictions/")
# df_with_pred.write.mode("overwrite").csv("output_predictions")

spark.stop()
