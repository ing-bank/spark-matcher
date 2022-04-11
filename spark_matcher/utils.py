# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import List

import numpy as np
import pandas as pd
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def get_most_frequent_words(sdf: DataFrame, col_name: str, min_df=2, top_n_words=1_000) -> pd.DataFrame:
    """
    Count word frequencies in a Spark dataframe `sdf` column named `col_name` and return a Pandas dataframe containing
    the document frequencies of the `top_n_words`. This function is intended to be used to create a list of stopwords.

    Args:
        sdf: Spark dataframe
        col_name: column name to get most frequent words from
        min_df: minimal document frequency for a word to be included as stopword, int for number and float for fraction
        top_n_words: number of most occurring words to include

    Returns:
        pandas dataframe containing most occurring words, their counts and document frequencies

    """
    sdf_col_splitted = sdf.withColumn(f'{col_name}_array', F.split(F.col(col_name), pattern=' '))
    word_count = sdf_col_splitted.select(F.explode(f'{col_name}_array').alias('words')).groupBy('words').count()
    doc_count = sdf.count()
    word_count = word_count.withColumn('df', F.col('count') / doc_count)
    if isinstance(min_df, int):
        min_count = min_df
    elif isinstance(min_df, float):
        min_count = np.ceil(min_df * doc_count)
    word_count_pdf = word_count.filter(F.col('count') >= min_count).sort(F.desc('df')).limit(top_n_words).toPandas()
    return word_count_pdf


def remove_stopwords(sdf: DataFrame, col_name: str, stopwords: List[str], case=False,
                     suffix: str = '_wo_stopwords') -> DataFrame:
    """
    Remove stopwords `stopwords` from a column `col_name` in a Spark dataframe `sdf`. The result will be written to a
    new column, named with the concatenation of `col_name` and `suffix`.

    Args:
        sdf: Spark dataframe
        col_name: column name to remove stopwords from
        stopwords: list of stopwords to remove
        case: whether to check for stopwords including lower- or uppercase
        suffix: suffix for the newly created column

    Returns:
        Spark dataframe with column added without stopwords

    """
    sdf = sdf.withColumn(f'{col_name}_array', F.split(F.col(col_name), pattern=' '))
    sw_remover = StopWordsRemover(inputCol=f'{col_name}_array', outputCol=f'{col_name}_array_wo_stopwords',
                                  stopWords=stopwords, caseSensitive=case)
    sdf = (sw_remover.transform(sdf)
           .withColumn(f'{col_name}{suffix}', F.concat_ws(' ', F.col(f'{col_name}_array_wo_stopwords')))
           .fillna({f'{col_name}{suffix}': ''})
           .drop(f'{col_name}_array', f'{col_name}_array_wo_stopwords')
           )
    return sdf
