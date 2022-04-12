# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import Tuple, Optional, Union
from pkg_resources import resource_filename

import pandas as pd

from pyspark.sql import SparkSession, DataFrame


def load_data(spark: SparkSession, kind: Optional[str] = 'voters') -> Union[Tuple[DataFrame, DataFrame], DataFrame]:
    """
    Load examples datasets to be used to experiment with `spark-matcher`. For matching problems, set `kind` to `voters`
    for North Carolina voter registry data or `library` for bibliography data. For deduplication problems, set `kind`
    to `stoxx50` for EuroStoxx 50 company names and addresses.

    Voter data:
        - provided by Prof. Erhard Rahm
          https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

    Library data:
        - DBLP bibliography, http://www.informatik.uni-trier.de/~ley/db/index.html
        - ACM Digital Library, http://portal.acm.org/portal.cfm

    Args:
        spark: Spark session
        kind: kind of data: `voters`, `library` or `stoxx50`

    Returns:
        two Spark dataframes for `voters` or `library`, a single dataframe for `stoxx50`

    """
    if kind == 'library':
        return _load_data_library(spark)
    if kind == 'voters':
        return _load_data_voters(spark)
    if kind == 'stoxx50':
        return _load_data_stoxx50(spark)
    else:
        raise ValueError('`kind` must be `library`, `voters` or `stoxx50`')


def _load_data_library(spark: SparkSession) -> Tuple[DataFrame, DataFrame]:
    """
    Load examples datasets to be used to experiment with `spark-matcher`. Two Spark dataframe are returned with the
    same columns:

    - DBLP bibliography, http://www.informatik.uni-trier.de/~ley/db/index.html
    - ACM Digital Library, http://portal.acm.org/portal.cfm

    Args:
        spark: Spark session

    Returns:
        Spark dataframe for DBLP data and a Spark dataframe for ACM data

    """
    file_path_acm = resource_filename('spark_matcher.data', 'acm.csv')
    file_path_dblp = resource_filename('spark_matcher.data', 'dblp.csv')
    acm_pdf = pd.read_csv(file_path_acm)
    dblp_pdf = pd.read_csv(file_path_dblp, encoding="ISO-8859-1")

    for col in acm_pdf.select_dtypes('object').columns:
        acm_pdf[col] = acm_pdf[col].fillna("")
    for col in dblp_pdf.select_dtypes('object').columns:
        dblp_pdf[col] = dblp_pdf[col].fillna("")

    acm_sdf = spark.createDataFrame(acm_pdf)
    dblp_sdf = spark.createDataFrame(dblp_pdf)
    return acm_sdf, dblp_sdf


def _load_data_voters(spark: SparkSession) -> Tuple[DataFrame, DataFrame]:
    """
    Voters data is based on the North Carolina voter registry and this dataset is provided by Prof. Erhard Rahm
    ('Comparative Evaluation of Distributed Clustering Schemes for Multi-source Entity Resolution'). Two Spark
    dataframe are returned with the same columns.

    Args:
        spark: Spark session

    Returns:
        two Spark dataframes containing voter data

    """
    file_path_voters_1 = resource_filename('spark_matcher.data', 'voters_1.csv')
    file_path_voters_2 = resource_filename('spark_matcher.data', 'voters_2.csv')
    voters_1_pdf = pd.read_csv(file_path_voters_1)
    voters_2_pdf = pd.read_csv(file_path_voters_2)

    voters_1_sdf = spark.createDataFrame(voters_1_pdf)
    voters_2_sdf = spark.createDataFrame(voters_2_pdf)
    return voters_1_sdf, voters_2_sdf


def _load_data_stoxx50(spark: SparkSession) -> DataFrame:
    """
    The Stoxx50 dataset contains a single column containing the concatenation of Eurostoxx 50 company names and
    addresses. This dataset is created by the developers of spark_matcher.

    Args:
        spark: Spark session

    Returns:
        Spark dataframe containing Eurostoxx 50 names and addresses

    """
    file_path_stoxx50 = resource_filename('spark_matcher.data', 'stoxx50.csv')
    stoxx50_pdf = pd.read_csv(file_path_stoxx50)

    stoxx50_sdf = spark.createDataFrame(stoxx50_pdf)
    return stoxx50_sdf
