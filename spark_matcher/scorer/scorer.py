# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import Union, Optional

import numpy as np
from pyspark.sql import SparkSession, types as T, functions as F, Column
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class Scorer:
    def __init__(self, spark_session: SparkSession, binary_clf: Optional[BaseEstimator] = None):
        self.spark_session = spark_session
        self.binary_clf = binary_clf
        if not self.binary_clf:
            self._create_default_clf()
        self.fitted_ = False

    def _create_default_clf(self) -> None:
        """
        This method creates a Sklearn Pipeline with a Standard Scaler and a Logistic Regression classifier.
        """
        self.binary_clf = (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight='balanced')
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Scorer':
        """
        This method fits a clf model on input data `X` nd the binary targets `y`.

        Args:
            X: training data
            y: training targets, containing binary values

        Returns:
            The object itself
        """

        if len(set(y)) == 1:  # in case active learning only resulted in labels from one class
            return self

        self.binary_clf.fit(X, y)
        self.fitted_ = True
        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        This method implements the code for predict_proba on a numpy array.
        This method is used to score all the pairs during training.
        """
        return self.binary_clf.predict_proba(X)

    def _predict_proba_spark(self, X: Column) -> Column:
        """
            This method implements the code for predict_proba on a spark column.
            This method is used to score all the pairs during inference time.
        """
        broadcasted_clf = self.spark_session.sparkContext.broadcast(self.binary_clf)

        @F.pandas_udf(T.FloatType())
        def _distributed_predict_proba(array):
            """
            This inner function defines the Pandas UDF for predict_proba on a spark cluster
            """
            return array.apply(lambda x: broadcasted_clf.value.predict_proba([x])[0][1])

        return _distributed_predict_proba(X)

    def predict_proba(self, X: Union[Column, np.ndarray]) -> Union[Column, np.ndarray]:
        """
        This method implements the abstract predict_proba method. It predicts the 'probabilities' of the target class
        for given input data `X`.

        Args:
            X: input data

        Returns:
            the predicted probabilities
        """
        if isinstance(X, Column):
            return self._predict_proba_spark(X)

        if isinstance(X, np.ndarray):
            return self._predict_proba(X)

        raise ValueError(f"{type(X)} is an unsupported datatype for X")
