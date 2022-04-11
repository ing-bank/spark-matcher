# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

import abc
import os

from pyspark.sql import SparkSession, DataFrame


class TableCheckpointer(abc.ABC):
    """
    Args:
        spark_session: a spark session
        database: a name of a database or storage system where the tables can be saved
        checkpoint_prefix: a prefix of the name that can be used to save tables
    """

    def __init__(self, spark_session: SparkSession, database: str, checkpoint_prefix: str = "checkpoint_spark_matcher"):
        self.spark_session = spark_session
        self.database = database
        self.checkpoint_prefix = checkpoint_prefix

    def __call__(self, sdf: DataFrame, checkpoint_name: str):
        return self.checkpoint_table(sdf, checkpoint_name)

    @abc.abstractmethod
    def checkpoint_table(self, sdf: DataFrame, checkpoint_name: str):
        pass


class HiveCheckpointer(TableCheckpointer):
    """
    Args:
        spark_session: a spark session
        database: a name of a database or storage system where the tables can be saved
        checkpoint_prefix: a prefix of the name that can be used to save tables
    """
    def __init__(self, spark_session: SparkSession, database: str, checkpoint_prefix: str = "checkpoint_spark_matcher"):
        super().__init__(spark_session, database, checkpoint_prefix)

    def checkpoint_table(self, sdf: DataFrame, checkpoint_name: str):
        """
        This method saves the input dataframes as checkpoints of the algorithm. This checkpointing can be
        used to store intermediary results that are needed throughout the algorithm. The tables are stored using the
        following name convention: `{checkpoint_prefix}_{checkpoint_name}`.

        Args:
            sdf: a Spark dataframe that needs to be saved as a checkpoint
            checkpoint_name: name of the table

        Returns:
            the same, unchanged, spark dataframe as the input dataframe. With the only difference that the
            dataframe is now read from disk as a checkpoint.
        """
        sdf.write.saveAsTable(f"{self.database}.{self.checkpoint_prefix}_{checkpoint_name}",
                              mode="overwrite")
        sdf = self.spark_session.table(f"{self.database}.{self.checkpoint_prefix}_{checkpoint_name}")
        return sdf


class ParquetCheckPointer(TableCheckpointer):
    """
    Args:
        spark_session: a spark session
        checkpoint_dir: directory where the tables can be saved
        checkpoint_prefix: a prefix of the name that can be used to save tables
    """
    def __init__(self, spark_session: SparkSession, checkpoint_dir: str,
                 checkpoint_prefix: str = "checkpoint_spark_matcher"):
        super().__init__(spark_session, checkpoint_dir, checkpoint_prefix)

    def checkpoint_table(self, sdf: DataFrame, checkpoint_name: str):
        """
        This method saves the input dataframes as checkpoints of the algorithm. This checkpointing can be
        used to store intermediary results that are needed throughout the algorithm. The tables are stored
        using the
        following name convention: `{checkpoint_prefix}_{checkpoint_name}`.

        Args:
            sdf: a Spark dataframe that needs to be saved as a checkpoint
            checkpoint_name: name of the table

        Returns:
            the same, unchanged, spark dataframe as the input dataframe. With the only difference that the
            dataframe is now read from disk as a checkpoint.
        """
        file_name = os.path.join(f'{self.database}', f'{self.checkpoint_prefix}_{checkpoint_name}')
        sdf.write.parquet(file_name, mode='overwrite')
        return self.spark_session.read.parquet(file_name)
