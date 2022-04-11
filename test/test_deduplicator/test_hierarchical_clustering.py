import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from pyspark.sql import SparkSession


def test__convert_data_to_adjacency_matrix():
    from spark_matcher.deduplicator.hierarchical_clustering import _convert_data_to_adjacency_matrix

    input_df = pd.DataFrame({
        'row_number_1': [1, 1],
        'row_number_2': [2, 3],
        'score': [0.8, 0.3],
        'component_id': [1, 1]})

    expected_indexes = np.array([1, 2, 3])
    expected_result = np.array([[0., 0.8, 0.3],
                                [0.8, 0., 0.],
                                [0.3, 0., 0.]])

    indexes, result = _convert_data_to_adjacency_matrix(input_df)

    np.testing.assert_array_equal(result, expected_result)
    np.testing.assert_array_equal(np.array(indexes), expected_indexes)


def test__get_condensed_distances():
    from spark_matcher.deduplicator.hierarchical_clustering import _get_condensed_distances

    input_df = pd.DataFrame({
        'row_number_1': [1, 1],
        'row_number_2': [2, 3],
        'score': [0.8, 0.3],
        'component_id': [1, 1]})

    expected_dict = {0: 1, 1: 2, 2: 3}
    expected_result = np.array([0.2, 0.7, 1.])

    result_dict, result = _get_condensed_distances(input_df)

    np.testing.assert_array_almost_equal_nulp(result, expected_result, nulp=2)
    np.testing.assert_array_equal(result_dict, expected_dict)


def test__convert_dedupe_result_to_pandas_dataframe(spark_session: SparkSession) -> None:
    from spark_matcher.deduplicator.hierarchical_clustering import _convert_dedupe_result_to_pandas_dataframe

    # case 1: with no empty data
    inputs = [
        ((1, 3), np.array([0.96, 0.96])),
        ((4, 7, 8), np.array([0.95, 0.95, 0.95])),
        ((5, 6), np.array([0.98, 0.98]))]
    component_id = 112233

    expected_df = pd.DataFrame(data={
        'row_number': [1, 3, 4, 7, 8, 5, 6],
        'entity_identifier': [f"{component_id}_0", f"{component_id}_0",
                              f"{component_id}_1", f"{component_id}_1", f"{component_id}_1",
                              f"{component_id}_2", f"{component_id}_2"]})
    result = _convert_dedupe_result_to_pandas_dataframe(inputs, component_id).reset_index(drop=True)

    assert_frame_equal(result, expected_df, check_dtype=False)

    # case 2: with empty data
    inputs = []
    component_id = 12345

    expected_df = pd.DataFrame(data={}, columns=['row_number', 'entity_identifier']).reset_index(drop=True)
    result = _convert_dedupe_result_to_pandas_dataframe(inputs, component_id).reset_index(drop=True)

    assert_frame_equal(result, expected_df, check_dtype=False)
