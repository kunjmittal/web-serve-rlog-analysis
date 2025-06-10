import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, count
import matplotlib.pyplot as plt
import pandas as pd
import os

# Initialize Spark
spark = SparkSession.builder.appName("WebServerLogAnalysis").getOrCreate()

# Read log file
log_path = "../data/access_log_sample.txt"
raw_logs = spark.read.text(log_path)

# Regular expression to parse common log format
log_pattern = r'(\S+) - - \[(.*?)\] "(.*?)" (\d{3}) (\d+)'

# Extract fields using regex
logs_df = raw_logs.select(
    regexp_extract('value', log_pattern, 1).alias('IP'),
    regexp_extract('value', log_pattern, 2).alias('Timestamp'),
    regexp_extract('value', log_pattern, 3).alias('Request'),
    regexp_extract('value', log_pattern, 4).alias('Status'),
    regexp_extract('value', log_pattern, 5).alias('Size')
)

# Further split the Request field into Method, Endpoint, and Protocol
logs_df = logs_df.withColumn("Method", regexp_extract("Request", r'(^\S+)', 1)) \
                 .withColumn("Endpoint", regexp_extract("Request", r'^\S+\s+(\S+)', 1)) \
                 .withColumn("Protocol", regexp_extract("Request", r'\s+HTTP/\d\.\d"', 0))

logs_df.show(truncate=False)

# Create output dir
os.makedirs("plots", exist_ok=True)

# ---- Visualization 1: Top Endpoints ----
endpoint_df = logs_df.groupBy("Endpoint") \
    .agg(count("*").alias("Hits")) \
    .orderBy(col("Hits").desc()) \
    .toPandas()

plt.figure(figsize=(10, 6))
plt.bar(endpoint_df['Endpoint'], endpoint_df['Hits'], color='skyblue')
plt.xlabel("Endpoint")
plt.ylabel("Hit Count")
plt.title("Top Requested Endpoints")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/top_endpoints.png")
print("📊 Saved: plots/top_endpoints.png")

# ---- Visualization 2: Status Code Distribution ----
status_df = logs_df.groupBy("Status") \
    .agg(count("*").alias("Count")) \
    .orderBy("Status") \
    .toPandas()

plt.figure(figsize=(6, 6))
plt.pie(status_df['Count'], labels=status_df['Status'], autopct='%1.1f%%', startangle=140)
plt.title("Status Code Distribution")
plt.savefig("plots/status_distribution.png")
print("📈 Saved: plots/status_distribution.png")
