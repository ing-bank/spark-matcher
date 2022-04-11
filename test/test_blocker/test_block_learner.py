import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.sql import functions as F, DataFrame

from spark_matcher.blocker.blocking_rules import BlockingRule
from spark_matcher.blocker.block_learner import BlockLearner


@pytest.fixture
def blocking_rule():
    class DummyBlockingRule(BlockingRule):
        def __repr__(self):
            return 'dummy_blocking_rule'

        def _blocking_rule(self, _):
            return F.lit('block_key')
    return DummyBlockingRule('blocking_column')


@pytest.fixture
def block_learner(blocking_rule, monkeypatch, table_checkpointer):
    monkeypatch.setattr(blocking_rule, "create_block_key", lambda _sdf: _sdf)
    return BlockLearner([blocking_rule, blocking_rule], 1.0, table_checkpointer)


def test__create_blocks(spark_session, blocking_rule, block_learner, monkeypatch):
    monkeypatch.setattr(block_learner, "cover_blocking_rules", [blocking_rule, blocking_rule])


    sdf = spark_session.createDataFrame(
        pd.DataFrame({
            'blocking_column': ['b', 'bb', 'bbb']
        })
    )

    result = block_learner._create_blocks(sdf)

    assert isinstance(result, list)
    assert isinstance(result[0], DataFrame)


def test__create_block_table(spark_session, block_learner):

    block_1 = spark_session.createDataFrame(
        pd.DataFrame({'column_1': [1, 2, 3], 'column_2': [4, 5, 6]})
    )
    block_2 = spark_session.createDataFrame(
        pd.DataFrame({'column_1': [7, 8, 9], 'column_2': [10, 11, 12]})
    )

    input_blocks = [block_1, block_2]

    expected_result = pd.DataFrame({
        'column_1': [1, 2, 3, 7, 8, 9], 'column_2': [4, 5, 6, 10, 11, 12]
    })

    result = (
        block_learner
        ._create_block_table(input_blocks)
        .toPandas()
        .sort_values(by='column_1')
        .reset_index(drop=True)
    )
    assert_frame_equal(result, expected_result)



def test__greedy_set_coverage(blocking_rule, block_learner):
    import copy

    full_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    block_learner.full_set = full_set
    block_learner.full_set_size = 10

    bl1 = copy.copy(blocking_rule)
    bl1.training_coverage = {0, 1, 2, 3}
    bl1.training_coverage_size = 4

    bl2 = copy.copy(blocking_rule)
    bl2.training_coverage = {3, 4, 5, 6, 7, 8}
    bl2.training_coverage_size = 6

    bl3 = copy.copy(blocking_rule)
    bl3.training_coverage = {3, 4, 5, 6, 7, 8}
    bl3.training_coverage_size = 5

    bl4 = copy.copy(blocking_rule)
    bl4.training_coverage = {8, 9}
    bl4.training_coverage_size = 2

    block_learner.blocking_rules = [bl1, bl2, bl3, bl4]

    # case 1: recall = 0.9
    block_learner.recall = 0.9
    block_learner._greedy_set_coverage()

    assert block_learner.cover_set == {0, 1, 2, 3, 4, 5, 6, 7, 8}

    # case 2: recall = 1.0
    block_learner.recall = 1.0
    block_learner._greedy_set_coverage()

    assert block_learner.cover_set == full_set

    # case 2: recall = 0.5
    block_learner.recall = 0.5
    block_learner._greedy_set_coverage()

    assert block_learner.cover_set == {3, 4, 5, 6, 7, 8}
