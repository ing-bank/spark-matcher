import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.sql import functions as F

from spark_matcher.blocker.blocking_rules import BlockingRule, FirstNChars, LastNChars, WholeField, FirstNWords, \
    FirstNLettersNoSpace, SortedIntegers, FirstInteger, LastInteger, LargestInteger, NLetterAbbreviation, \
    FirstNCharsLastWord, FirstNCharactersFirstTokenSorted


@pytest.fixture
def blocking_rule():
    class DummyBlockingRule(BlockingRule):
        def __repr__(self):
            return 'dummy_blocking_rule'

        def _blocking_rule(self, _):
            return F.lit('block_key')
    return DummyBlockingRule('blocking_column')


def test_create_block_key(spark_session, blocking_rule):
    input_sdf = spark_session.createDataFrame(
        pd.DataFrame({
            'sort_column': [1, 2, 3],
            'blocking_column': ['a', 'aa', 'aaa']}))

    expected_result = pd.DataFrame({
        'sort_column': [1, 2, 3],
        'blocking_column': ['a', 'aa', 'aaa'],
        'block_key': ['dummy_blocking_rule:block_key', 'dummy_blocking_rule:block_key', 'dummy_blocking_rule:block_key']
    })

    result = (
        blocking_rule
        .create_block_key(input_sdf)
        .toPandas()
        .sort_values(by='sort_column')
        .reset_index(drop=True)
    )
    assert_frame_equal(result, expected_result)


def test__create_training_block_keys(spark_session, blocking_rule):
    input_sdf = spark_session.createDataFrame(
        pd.DataFrame({
            'sort_column': [1, 2, 3],
            'blocking_column_1': ['a', 'aa', 'aaa'],
            'blocking_column_2': ['b', 'bb', 'bbb']}))

    expected_result = pd.DataFrame({
        'sort_column': [1, 2, 3],
        'blocking_column_1': ['a', 'aa', 'aaa'],
        'blocking_column_2': ['b', 'bb', 'bbb'],
        'block_key_1': ['dummy_blocking_rule:block_key', 'dummy_blocking_rule:block_key',
                        'dummy_blocking_rule:block_key'],
        'block_key_2': ['dummy_blocking_rule:block_key', 'dummy_blocking_rule:block_key',
                        'dummy_blocking_rule:block_key']
    })

    result = (
        blocking_rule
        ._create_training_block_keys(input_sdf)
        .toPandas()
        .sort_values(by='sort_column')
        .reset_index(drop=True)
    )
    assert_frame_equal(result, expected_result)


def test__compare_and_filter_keys(spark_session, blocking_rule):
    input_sdf = spark_session.createDataFrame(
        pd.DataFrame({
            'sort_column': [1, 2, 3, 4, 5, 6],
            'block_key_1': ['blocking_rule:a', 'blocking_rule:a:a', 'blocking_rule:a:', 'blocking_rule:aaa',
                            'blocking_rule:', 'blocking_rule:'],
            'block_key_2': ['blocking_rule:a', 'blocking_rule:a:a', 'blocking_rule:a:', 'blocking_rule:bbb',
                            'blocking_rule:', 'blocking_rule:bb']}))

    expected_result = pd.DataFrame({
            'sort_column': [1, 2, 3],
            'block_key_1': ['blocking_rule:a', 'blocking_rule:a:a', 'blocking_rule:a:'],
            'block_key_2': ['blocking_rule:a', 'blocking_rule:a:a', 'blocking_rule:a:']})

    result = (
        blocking_rule
        ._compare_and_filter_keys(input_sdf)
        .toPandas()
        .sort_values(by='sort_column')
        .reset_index(drop=True)
    )
    print(input_sdf.show())
    print('res')
    print(result)
    print('exp')
    print(expected_result)
    assert_frame_equal(result, expected_result)


def test_calculate_training_set_coverage(spark_session, blocking_rule, monkeypatch):
    # monkey_patch the inner functions of `calculate_training_set_coverage`, since they are tested separately
    monkeypatch.setattr(blocking_rule, "_create_training_block_keys", lambda _sdf: _sdf)
    monkeypatch.setattr(blocking_rule, "_compare_and_filter_keys", lambda _sdf: _sdf)

    row_ids = [0, 1, 2, 3, 4, 5]
    input_sdf = spark_session.createDataFrame(
        pd.DataFrame({'row_id': row_ids}))

    expected_result = set(row_ids)

    blocking_rule.calculate_training_set_coverage(input_sdf)

    assert blocking_rule.training_coverage == expected_result
    assert blocking_rule.training_coverage_size == len(expected_result)


@pytest.mark.parametrize('n', [2, 3])
def test_firstnchars(spark_session, n):
    if n == 2:
        expected_result_pdf = pd.DataFrame({'name': ['aa', 'spark', 'aa bb'],
                                            'block_key': ["first_2_characters_name:aa",
                                                          "first_2_characters_name:sp",
                                                          "first_2_characters_name:aa"]})
    elif n == 3:
        expected_result_pdf = pd.DataFrame({'name': ['aa', 'spark', 'aa bb'],
                                            'block_key': [None,
                                                          "first_3_characters_name:spa",
                                                          "first_3_characters_name:aa "]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = FirstNChars('name', n).create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


@pytest.mark.parametrize('n', [2, 3])
def test_firstncharslastword(spark_session, n):
    if n == 2:
        expected_result_pdf = pd.DataFrame({'name': ['aa', 'aa spark', 'aa bb'],
                                            'block_key': ["first_2_characters_last_word_name:aa",
                                                          "first_2_characters_last_word_name:sp",
                                                          "first_2_characters_last_word_name:bb"]})
    elif n == 3:
        expected_result_pdf = pd.DataFrame({'name': ['aa', 'aa spark', 'aa bbb'],
                                            'block_key': [None,
                                                          "first_3_characters_last_word_name:spa",
                                                          "first_3_characters_last_word_name:bbb"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = FirstNCharsLastWord('name', n).create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


@pytest.mark.parametrize('n', [2, 3])
def test_firstncharactersfirsttokensorted(spark_session, n):
    if n == 2:
        expected_result_pdf = pd.DataFrame({'name': ['aa', 'spark aa', 'aa bb', 'a bb'],
                                            'block_key': ["first_2_characters_first_token_sorted_name:aa",
                                                          "first_2_characters_first_token_sorted_name:aa",
                                                          "first_2_characters_first_token_sorted_name:aa",
                                                          "first_2_characters_first_token_sorted_name:bb"]})
    elif n == 3:
        expected_result_pdf = pd.DataFrame({'name': ['aa', 'spark aaa', 'bbb aa'],
                                            'block_key': [None,
                                                          "first_3_characters_first_token_sorted_name:aaa",
                                                          "first_3_characters_first_token_sorted_name:bbb"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = FirstNCharactersFirstTokenSorted('name', n).create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


@pytest.mark.parametrize('n', [2, 3])
def test_lastnchars(spark_session, n):
    if n == 2:
        expected_result_pdf = pd.DataFrame({'name': ['aa', 'spark', 'aa bb'],
                                            'block_key': ["last_2_characters_name:aa",
                                                          "last_2_characters_name:rk",
                                                          "last_2_characters_name:bb"]})
    elif n == 3:
        expected_result_pdf = pd.DataFrame({'name': ['aa', 'spark', 'aa bb'],
                                            'block_key': [None,
                                                          "last_3_characters_name:ark",
                                                          "last_3_characters_name: bb"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = LastNChars('name', n).create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


def test_wholefield(spark_session):
    expected_result_pdf = pd.DataFrame({'name': ['python', 'python pyspark'],
                                        'block_key': [f"whole_field_name:python",
                                                      f"whole_field_name:python pyspark"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = WholeField('name').create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


@pytest.mark.parametrize('n', [2, 3])
def test_firstnwords(spark_session, n):
    if n == 2:
        expected_result_pdf = pd.DataFrame({'name': ['python', 'python pyspark'],
                                            'block_key': [None,
                                                          "first_2_words_name:python pyspark"]})
    elif n == 3:
        expected_result_pdf = pd.DataFrame({'name': ['python', 'python py spark'],
                                            'block_key': [None, "first_3_words_name:python py spark"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = FirstNWords('name', n).create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


@pytest.mark.parametrize('n', [2, 3])
def test_firstnlettersnospace(spark_session, n):
    if n == 2:
        expected_result_pdf = pd.DataFrame({'name': ['python', 'p ython'],
                                            'block_key': ["first_2_letters_name_no_space:py",
                                                          "first_2_letters_name_no_space:py"]})
    elif n == 3:
        expected_result_pdf = pd.DataFrame({'name': ['p y', 'python', 'p y thon'],
                                            'block_key': [None,
                                                          "first_3_letters_name_no_space:pyt",
                                                          "first_3_letters_name_no_space:pyt"]})

        sdf = spark_session.createDataFrame(expected_result_pdf)
        result_pdf = FirstNLettersNoSpace('name', n).create_block_key(sdf.select('name')).toPandas()
        assert_frame_equal(expected_result_pdf, result_pdf)


def test_sortedintegers(spark_session):
    expected_result_pdf = pd.DataFrame({'name': ['python', 'python 2 1'],
                                        'block_key': [None,
                                                      "sorted_integers_name:1 2"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = SortedIntegers('name').create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


def test_firstinteger(spark_session):
    expected_result_pdf = pd.DataFrame({'name': ['python', 'python 2 1'],
                                        'block_key': [None,
                                                      "first_integer_name:2"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = FirstInteger('name').create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


def test_lastinteger(spark_session):
    expected_result_pdf = pd.DataFrame({'name': ['python', 'python 2 1'],
                                        'block_key': [None,
                                                      "last_integer_name:1"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = LastInteger('name').create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


def test_largestinteger(spark_session):
    expected_result_pdf = pd.DataFrame({'name': ['python', 'python1', 'python 2 1'],
                                        'block_key': [None,
                                                      'largest_integer_name:1',
                                                      "largest_integer_name:2"]})

    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = LargestInteger('name').create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)


@pytest.mark.parametrize('n', [2, 3])
def test_nletterabbreviation(spark_session, n):
    if n == 2:
        expected_result_pdf = pd.DataFrame({'name': ['python', 'python pyspark'],
                                            'block_key': [None,
                                                          "2_letter_abbreviation_name:pp"]})
    elif n == 3:
        expected_result_pdf = pd.DataFrame({'name': ['python', 'python pyspark', 'python apache pyspark'],
                                            'block_key': [None,
                                                          None,
                                                          "3_letter_abbreviation_name:pap"]})
    sdf = spark_session.createDataFrame(expected_result_pdf)
    result_pdf = NLetterAbbreviation('name', n).create_block_key(sdf.select('name')).toPandas()
    assert_frame_equal(expected_result_pdf, result_pdf)
