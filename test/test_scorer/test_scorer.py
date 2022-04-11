import pytest
import pandas as pd
import numpy as np
from pyspark.sql import Column
from sklearn.base import BaseEstimator

from spark_matcher.scorer.scorer import Scorer


@pytest.fixture
def scorer(spark_session):
    class DummyScorer(Scorer):
        def __init__(self, spark_session=spark_session):
            super().__init__(spark_session)

        def _predict_proba(self, X):
            return np.array([[0, 1]])

        def _predict_proba_spark(self, X):
            return spark_session.createDataFrame(pd.DataFrame({'1': [1]}))['1']

    return DummyScorer()


def test_fit(scorer):
    # case 1, the scorer should be able to be 'fitted' without execptions even if there is only one class:
    X = np.array([[1, 2, 3]])
    y = pd.Series([0])
    scorer.fit(X, y)

    # case 2:
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = pd.Series([0, 1])
    scorer.fit(X, y)


def test_predict_proba(spark_session, scorer):
    X = np.ndarray([0])
    preds = scorer.predict_proba(X)
    assert isinstance(preds, np.ndarray)

    X = spark_session.createDataFrame(pd.DataFrame({'c': [0]}))
    preds = scorer.predict_proba(X['c'])
    assert isinstance(preds, Column)

    X = pd.DataFrame({})
    with pytest.raises(ValueError) as e:
        scorer.predict_proba(X)
    assert f"{type(X)} is an unsupported datatype for X" == str(e.value)


def test__create_default_clf(scorer):
    clf = scorer.binary_clf
    assert isinstance(clf, BaseEstimator)
    assert hasattr(clf, 'fit')
    assert hasattr(clf, 'predict_proba')