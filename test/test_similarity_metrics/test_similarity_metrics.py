import pandas as pd
from pandas.testing import assert_series_equal
from thefuzz.fuzz import token_set_ratio

from spark_matcher.similarity_metrics import SimilarityMetrics


def test__distance_measure_pandas():
    df = pd.DataFrame({'name_1': ['aa', 'bb', 'cc'],
                       'name_2': ['aa', 'bb c', 'dd'],
                       'similarity_metrics': [100, 100, 0]
                       })
    strings_1 = df['name_1'].tolist()
    strings_2 = df['name_2'].tolist()
    result = SimilarityMetrics._distance_measure_pandas(strings_1, strings_2, token_set_ratio)
    assert_series_equal(result, df['similarity_metrics'], check_names=False)
