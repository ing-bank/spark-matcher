import os
from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from thefuzz.fuzz import token_set_ratio

from spark_matcher.blocker.blocking_rules import BlockingRule
from spark_matcher.matching_base.matching_base import MatchingBase


def create_fake_string(length=2):
    return "".join(np.random.choice(list(ascii_lowercase), size=length))


def create_fake_df(size=10):
    return pd.DataFrame.from_dict({'name': [create_fake_string(1) + ' ' + create_fake_string(2) for _ in range(size)],
                                   'address': [create_fake_string(2) + ' ' + create_fake_string(3) for _ in range(size)]})


@pytest.fixture()
def matching_base(spark_session, table_checkpointer):
    return MatchingBase(spark_session, table_checkpointer=table_checkpointer, col_names=['name', 'address'])


def test__create_train_pairs_table(spark_session, matching_base):
    n_1, n_2, n_train_samples, n_perfect_train_matches = (1000, 100, 5000, 1)
    sdf_1 = spark_session.createDataFrame(create_fake_df(size=n_1))
    sdf_2 = spark_session.createDataFrame(create_fake_df(size=n_2))

    matching_base.n_train_samples = n_train_samples
    matching_base.n_perfect_matches = n_perfect_train_matches

    result = matching_base._create_train_pairs_table(sdf_1, sdf_2)

    assert set(result.columns) == {'name_2', 'address_2', 'name_1', 'address_1', 'perfect_train_match'}
    assert result.count() == pytest.approx(n_train_samples, abs=n_perfect_train_matches)


def test_default_blocking_rules(matching_base):
    assert isinstance(matching_base.blocking_rules, list)
    assert isinstance(matching_base.blocking_rules[0], BlockingRule)


def test_load_save(spark_session, tmpdir, matching_base, table_checkpointer):
    def my_metric(x, y):
        return float(x == y)

    matching_base.field_info = {'name': [token_set_ratio, my_metric]}
    matching_base.n_perfect_train_matches = 5
    matching_base.save(os.path.join(tmpdir, 'matcher.pkl'))

    myMatcherLoaded = MatchingBase(spark_session, table_checkpointer=table_checkpointer, col_names=['nothing'])
    myMatcherLoaded.load(os.path.join(tmpdir, 'matcher.pkl'))
    setattr(table_checkpointer, 'spark_session', spark_session)  # needed to be able to continue with other unit tests

    assert matching_base.col_names == myMatcherLoaded.col_names
    assert [x.__name__ for x in matching_base.field_info['name']] == [x.__name__ for x in
                                                                  myMatcherLoaded.field_info['name']]
    assert matching_base.n_perfect_train_matches == myMatcherLoaded.n_perfect_train_matches


@pytest.mark.parametrize('suffix', [1, 2, 3])
def test__add_suffix_to_col_names(spark_session, suffix, matching_base):
    input_df = pd.DataFrame({
        'name': ['John Doe', 'Jane Doe', 'Chris Rock', 'Jack Sparrow'],
        'address': ['Square 1', 'Square 2', 'Main street', 'Harbour 123'],
        'block_key': ['1', '2', '3', '3']
    })

    result = (
        matching_base
        ._add_suffix_to_col_names(spark_session.createDataFrame(input_df), suffix)
        .toPandas()
        .reset_index(drop=True)
    )
    assert_frame_equal(result, input_df.rename(columns={'name': f'name_{suffix}', 'address': f'address_{suffix}'}))
