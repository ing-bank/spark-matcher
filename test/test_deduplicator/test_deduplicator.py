import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.sql import functions as F

from spark_matcher.deduplicator import Deduplicator


@pytest.fixture()
def deduplicator(spark_session):
    return Deduplicator(spark_session, col_names=[], checkpoint_dir='mock_db')


def test__create_predict_pairs_table(spark_session, deduplicator):
    deduplicator.col_names = ['name', 'address']

    input_df = pd.DataFrame(
        {
            'name': ['John Doe', 'Jane Doe', 'Daniel Jacks', 'Jack Sparrow', 'Donald Trump'],
            'address': ['1 Square', '2 Square', '3 Main street', '4 Harbour', '5 white house'],
            'block_key': ['J', 'J', 'D', 'J', 'D'],
            'row_number': [1, 2, 3, 4, 5],
        }
    )

    expected_df = pd.DataFrame({
        'block_key': ['J', 'J', 'J', 'D'],
        'name_1': ['John Doe', 'John Doe', 'Jane Doe', 'Daniel Jacks'],
        'address_1': ['1 Square', '1 Square', '2 Square', '3 Main street'],
        'row_number_1': [1, 1, 2, 3],
        'name_2': ['Jane Doe', 'Jack Sparrow', 'Jack Sparrow', 'Donald Trump'],
        'address_2': ['2 Square', '4 Harbour', '4 Harbour', '5 white house'],
        'row_number_2': [2, 4, 4, 5]
    })

    result = (
        deduplicator
        ._create_predict_pairs_table(spark_session.createDataFrame(input_df))
        .toPandas()
        .sort_values(by=['address_1', 'address_2'])
        .reset_index(drop=True)
    )
    assert_frame_equal(result, expected_df)


def test__add_singletons_entity_identifiers(spark_session, deduplicator) -> None:
    df = pd.DataFrame(
        data={
            "entity_identifier": [0, 1, None, 2, 3, 4, 5, 6],
            "row_number": [10, 11, 12, 13, 14, 15, 16, 17]})

    input_data = (
        spark_session
            .createDataFrame(df)
            .withColumn('entity_identifier',
                        F.when(F.isnan('entity_identifier'), F.lit(None)).otherwise(F.col('entity_identifier')))
            .repartition(6))

    expected_df = pd.DataFrame(
        data={
            "row_number": [10, 11, 12, 13, 14, 15, 16, 17],
            "entity_identifier": [0.0, 1.0, 7.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    result = (deduplicator._add_singletons_entity_identifiers(input_data)
              .toPandas()
              .sort_values(by='row_number')
              .reset_index(drop=True))
    assert_frame_equal(result, expected_df, check_dtype=False)
