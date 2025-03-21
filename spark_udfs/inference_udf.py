from pyspark.sql import SparkSession
from spark_udfs.inference_udf import InferenceUDF

# Create a Spark session
spark = SparkSession.builder.appName("InferenceUDF").getOrCreate()

# # Create a sample DataFrame
# data = [("John", 25), ("Jane", 22), ("Doe", 30)]
# columns = ["name", "age"]
# df = spark.createDataFrame(data, columns)

# # Create an instance of the InferenceUDF class
# inference_udf = InferenceUDF()

# # Register the UDF with the Spark session
# spark.udf.register("inference_udf", inference_udf, returnType=inference_udf.return_type)

# # Use the UDF in a SQL query
# df.createOrReplaceTempView("people")

# result = spark.sql("SELECT name, age, inference_udf(age) as prediction FROM people")
# result.show()

# Stop the Spark session
spark.stop()
