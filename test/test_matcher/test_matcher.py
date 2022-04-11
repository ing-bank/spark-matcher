import pytest
import pandas as pd
from thefuzz.fuzz import ratio
from pyspark.sql import DataFrame

from spark_matcher.blocker.block_learner import BlockLearner
from spark_matcher.blocker.blocking_rules import FirstNChars
from spark_matcher.scorer.scorer import Scorer
from spark_matcher.matcher import Matcher


@pytest.fixture
def matcher(spark_session, table_checkpointer):
    block_name = FirstNChars('name', 3)
    block_address = FirstNChars('address', 3)
    blocklearner = BlockLearner([block_name, block_address], recall=1, table_checkpointer=table_checkpointer)
    blocklearner.fitted = True
    blocklearner.cover_blocking_rules = [block_name, block_address]

    myScorer = Scorer(spark_session)
    fit_df = pd.DataFrame({'similarity_metrics': [[100, 80], [50, 50]], 'y': [1, 0]})
    myScorer.fit(fit_df['similarity_metrics'].values.tolist(), fit_df['y'])

    myMatcher = Matcher(spark_session, table_checkpointer=table_checkpointer,
                        field_info={'name': [ratio], 'address': [ratio]})
    myMatcher.blocker = blocklearner
    myMatcher.scorer = myScorer
    myMatcher.fitted_ = True

    return myMatcher


def test__create_predict_pairs_table(spark_session, table_checkpointer):
    sdf_1_blocked = spark_session.createDataFrame(
        pd.DataFrame({'name': ['frits', 'stan', 'ahmet'], 'address': ['dam 1', 'leidseplein 2', 'rokin 3'],
                      'block_key': ['fr', 'st', 'ah']}))
    sdf_2_blocked = spark_session.createDataFrame(
        pd.DataFrame({'name': ['frits', 'fred', 'glenn'],
                      'address': ['amsterdam', 'waterlooplein 4', 'rembrandtplein 5'],
                      'block_key': ['fr', 'fr', 'gl']}))
    myMatcher = Matcher(spark_session, table_checkpointer=table_checkpointer, col_names=['name', 'address'])
    result = myMatcher._create_predict_pairs_table(sdf_1_blocked, sdf_2_blocked)
    assert isinstance(result, DataFrame)
    assert result.count() == 2
    assert result.select('block_key').drop_duplicates().count() == 1
    assert set(result.columns) == {'block_key', 'name_2', 'address_2', 'name_1', 'address_1'}

def test_predict(spark_session, matcher):
    sdf_1 = spark_session.createDataFrame(pd.DataFrame({'name': ['frits', 'stan', 'ahmet', 'ahmet', 'ahmet'],
                                                        'address': ['damrak', 'leidseplein', 'waterlooplein',
                                                                    'amstel', 'amstel 3']}))

    sdf_2 = spark_session.createDataFrame(
        pd.DataFrame({'name': ['frits h', 'stan l', 'bayraktar', 'ahmet', 'ahmet'],
                      'address': ['damrak 1', 'leidseplein 2', 'amstel 3', 'amstel 3', 'waterlooplein 12324']}))

    result = matcher.predict(sdf_1, sdf_2)
    assert result
