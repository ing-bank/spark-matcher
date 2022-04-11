import pytest

import pandas as pd


@pytest.mark.parametrize('min_df', [1, 2, 0.2])
def test_get_most_frequent_words(spark_session, min_df):
    from spark_matcher.utils import get_most_frequent_words

    sdf = spark_session.createDataFrame(pd.DataFrame({'name': ['Company A ltd', 'Company B ltd', 'Company C']}))

    if (min_df == 1) or (min_df == 0.2):
        expected_result = pd.DataFrame({'words': ['Company', 'ltd', 'A', 'B', 'C'],
                                        'count': [3, 2, 1, 1, 1],
                                        'df': [1, 2 / 3, 1 / 3, 1 / 3, 1 / 3]})
    elif min_df == 2:
        expected_result = pd.DataFrame({'words': ['Company', 'ltd'],
                                        'count': [3, 2],
                                        'df': [1, 2 / 3]})

    result = get_most_frequent_words(sdf, col_name='name', min_df=min_df, top_n_words=10).sort_values(['count', \
                                    'words'], ascending=[False, True]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_result, result)


def test_remove_stop_words(spark_session):
    from spark_matcher.utils import remove_stopwords

    stopwords = ['ltd', 'bv']

    expected_sdf = spark_session.createDataFrame(
        pd.DataFrame({'name': ['Company A ltd', 'Company B ltd', 'Company C bv', 'Company D'],
                      'name_wo_stopwords': ['Company A', 'Company B', 'Company C', 'Company D']}))
    input_cols = ['name']

    result = remove_stopwords(expected_sdf.select(*input_cols), col_name='name', stopwords=stopwords)
    pd.testing.assert_frame_equal(result.toPandas(), expected_sdf.toPandas())
