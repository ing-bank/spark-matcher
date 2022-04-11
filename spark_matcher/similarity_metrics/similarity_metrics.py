# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import List, Callable, Dict

import pandas as pd

from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as F
from pyspark.sql import types as T


class SimilarityMetrics:
    """
    Class to calculate similarity metrics for pairs of records. The `field_info` dict contains column names as keys
    and lists of similarity functions as values. E.g.

    field_info = {'name': [token_set_ratio, token_sort_ratio],
                  'postcode': [ratio]}

    where `token_set_ratio`, `token_sort_ratio` and `ratio` are string similarity functions that take two strings
    as arguments and return a numeric value

    Attributes:
        field_info: dict containing column names as keys and lists of similarity functions as values
    """
    def __init__(self, field_info: Dict):
        self.field_info = field_info

    @staticmethod
    def _distance_measure_pandas(strings_1: List[str], strings_2: List[str], func: Callable) -> pd.Series:
        """
        Helper function to apply a string similarity metric to two arrays of strings. To be used in a Pandas UDF.

        Args:
            strings_1: array containing strings
            strings_2: array containing strings
            func: string similarity function to be applied

        Returns:
            Pandas series containing string similarities

        """
        df = pd.DataFrame({'x': strings_1, 'y': strings_2})
        return df.apply(lambda row: func(row['x'], row['y']), axis=1)

    @staticmethod
    def _create_similarity_metric_udf(similarity_metric_function: Callable):
        """
        Function that created Pandas UDF for a given similarity_metric_function

        Args:
            similarity_metric_function: function that takes two strings and returns a number

        Returns:
            Pandas UDF
        """

        @F.pandas_udf(T.FloatType())
        def similarity_udf(strings_1: pd.Series, strings_2: pd.Series) -> pd.Series:
            # some similarity metrics cannot deal with empty strings, therefore these are replaced with " "
            strings_1 = [x if x != "" else " " for x in strings_1]
            strings_2 = [x if x != "" else " " for x in strings_2]
            return SimilarityMetrics._distance_measure_pandas(strings_1, strings_2, similarity_metric_function)

        return similarity_udf

    def _apply_distance_metrics(self) -> Column:
        """
        Function to apply all distance metrics in the right order to string pairs and returns them in an array. This
        array is used as input to the (logistic regression) scoring function.

        Returns:
            array with the string distance metrics

        """
        distance_metrics_list = []
        for field_name in self.field_info.keys():
            field_name_1, field_name_2 = field_name + "_1", field_name + "_2"
            for similarity_function in self.field_info[field_name]:
                similarity_metric = (
                    SimilarityMetrics._create_similarity_metric_udf(similarity_function)(F.col(field_name_1),
                                                                                         F.col(field_name_2)))
                distance_metrics_list.append(similarity_metric)

        array = F.array(*distance_metrics_list)

        return array

    def transform(self, pairs_table: DataFrame) -> DataFrame:
        """
        Method to apply similarity metrics to pairs table. Method makes use of method dispatching to facilitate both
        Pandas and Spark dataframes

        Args:
            pairs_table: Spark or Pandas dataframe containing pairs table

        Returns:
            Pandas or Spark dataframe with pairs table and newly created `similarity_metrics` column

        """
        return pairs_table.withColumn('similarity_metrics', self._apply_distance_metrics())
