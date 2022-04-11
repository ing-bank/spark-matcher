import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from string import ascii_lowercase

from spark_matcher.sampler.training_sampler import HashSampler, RandomSampler


def create_fake_string(length=2):
    return "".join(np.random.choice(list(ascii_lowercase), size=length))


def create_fake_df(size=10):
    return pd.DataFrame.from_dict({'name': [create_fake_string(1) + ' ' + create_fake_string(2) for _ in range(size)],
                                'surname': [create_fake_string(2) + ' ' + create_fake_string(3) for _ in range(size)]})


@pytest.mark.parametrize('test_case', [(100_000, 10, 10_000), (100, 100_000, 10_000), (100_000, 100_000, 10_000)])
def test__create_random_pairs_table_matcher(spark_session, test_case, table_checkpointer):
    n_1, n_2, n_train_samples = test_case
    sdf_1 = spark_session.createDataFrame(create_fake_df(size=n_1))
    sdf_2 = spark_session.createDataFrame(create_fake_df(size=n_2))

    rSampler = RandomSampler(table_checkpointer, col_names=['name', 'surname'], n_train_samples=n_train_samples)

    result = rSampler._create_pairs_table_matcher(sdf_1, sdf_2)

    assert set(result.columns) == {'name_2', 'surname_2', 'name_1', 'surname_1'}
    assert result.count() == pytest.approx(n_train_samples, abs=1)
    if n_1 > n_2:
        assert result.select('name_1', 'surname_1').drop_duplicates().count() == pytest.approx(
            n_train_samples / n_2, abs=1)
        assert result.select('name_2', 'surname_2').drop_duplicates().count() == n_2
    elif n_1 < n_2:
        assert result.select('name_1', 'surname_1').drop_duplicates().count() == pytest.approx(n_1, abs=1)
        assert result.select('name_2', 'surname_2').drop_duplicates().count() == pytest.approx(
            n_train_samples / n_1, abs=1)
    elif n_1 == n_2:
        assert result.select('name_2', 'surname_2').drop_duplicates().count() == pytest.approx(
            int(n_train_samples ** 0.5), abs=1)
        assert result.select('name_1', 'surname_1').drop_duplicates().count() == pytest.approx(
            int(n_train_samples ** 0.5), abs=1)


@pytest.mark.parametrize('test_case', [(100_000, 10_000), (1_000, 10_000), (1_000, 1_000)])
def test__create_random_pairs_table_deduplicator(spark_session, test_case, table_checkpointer):
    n_rows, n_train_samples = test_case
    sdf = spark_session.createDataFrame(create_fake_df(size=n_rows))

    rSampler = RandomSampler(table_checkpointer, col_names=['name', 'surname'], n_train_samples=n_train_samples)

    result = rSampler._create_pairs_table_deduplicator(sdf)

    assert set(result.columns) == {'name_2', 'surname_2', 'name_1', 'surname_1'}
    assert result.count() == n_train_samples


def test__create_hashed_pairs_table_matcher(spark_session, table_checkpointer):
    n_1, n_2, n_train_samples = (1000, 1000, 1000)
    sdf_1 = spark_session.createDataFrame(create_fake_df(size=n_1))
    sdf_2 = spark_session.createDataFrame(create_fake_df(size=n_2))

    hSampler = HashSampler(table_checkpointer, col_names=['name', 'surname'], n_train_samples=n_train_samples, threshold=1)

    result = hSampler.create_pairs_table(sdf_1, sdf_2)

    assert set(result.columns) == {'name_2', 'surname_2', 'name_1', 'surname_1'}
    assert result.count() == n_train_samples


def test__create_hashed_pairs_table_deduplicator(spark_session, table_checkpointer):
    n_1, n_train_samples = (1000, 1000)
    sdf_1 = spark_session.createDataFrame(create_fake_df(size=n_1))

    hSampler = HashSampler(table_checkpointer, col_names=['name', 'surname'], n_train_samples=n_train_samples, threshold=1)

    result = hSampler.create_pairs_table(sdf_1)

    assert set(result.columns) == {'name_2', 'surname_2', 'name_1', 'surname_1'}
    assert result.count() == n_train_samples


def test__vectorize_matcher(spark_session, table_checkpointer):
    input_sdf_1 = spark_session.createDataFrame(
        pd.DataFrame({
            'name': ['aa bb cc', 'cc dd'],
            'surname': ['xx yy', 'yy zz xx']
        }))

    input_sdf_2 = spark_session.createDataFrame(
        pd.DataFrame({
            'name': ['bb cc'],
            'surname': ['yy']
        }))

    hSampler = HashSampler(table_checkpointer, col_names=['name', 'surname'], n_train_samples=999)

    # set max_df to a large value because otherwise nothing would be tokenized
    result_sdf_1, result_sdf_2 = hSampler._vectorize(['name', 'surname'], input_sdf_1, input_sdf_2, max_df=10)

    result_df_1 = result_sdf_1.toPandas()
    result_df_2 = result_sdf_2.toPandas()

    assert set(result_df_1.columns) == {'name', 'surname', 'features'}
    assert set(result_df_2.columns) == {'name', 'surname', 'features'}
    assert result_df_1.shape[0] == 2
    assert result_df_2.shape[0] == 1
    assert len(result_df_1.features[0]) == 4  # note that not all tokens occur at least twice (minDF=2)
    assert len(result_df_2.features[0]) == 4  # note that not all tokens occur at least twice (minDF=2)


def test__vectorize_deduplicator(spark_session, table_checkpointer):
    input_sdf = spark_session.createDataFrame(
        pd.DataFrame({
            'name': ['aa bb cc', 'cc dd', 'bb cc'],
            'surname': ['xx yy', 'yy zz xx', 'yy zz']
        }))

    hSampler = HashSampler(table_checkpointer, col_names=['name', 'surname'], n_train_samples=999)

    # set max_df to a large value because otherwise nothing would be tokenized
    result_sdf = hSampler._vectorize(['name', 'surname'], input_sdf, max_df=10)

    result_df = result_sdf.toPandas()

    assert set(result_df.columns) == {'name', 'surname', 'features'}
    assert result_df.shape[0] == 3
    assert len(result_df.features[0]) == 5  # note that not all tokens occur at least twice (minDF=2)
