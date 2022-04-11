import pytest
from pyspark.sql import SparkSession

from spark_matcher.table_checkpointer import ParquetCheckPointer


@pytest.fixture(scope="session")
def spark_session():
    spark = (SparkSession
             .builder
             .appName(str(__file__))
             .getOrCreate()
             )
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def table_checkpointer(spark_session):
    return ParquetCheckPointer(spark_session, 'temp_database')
