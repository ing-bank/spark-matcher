from typing import List, Union
import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from pyspark.sql import DataFrame
from sklearn.base import BaseEstimator
from spark_matcher.activelearner.active_learner_base import ActiveLearnerBase


class ConfidenceLearner(ActiveLearnerBase):
    """
    Class to train a string matching model using active learning.
    Attributes:
        col_names: column names used for matching
        scorer: the scorer to be used in the active learning loop
        min_nr_samples: minimum number of responses required before classifier convergence is tested
        uncertainty_threshold: threshold on the uncertainty of the classifier during active learning,
            used for determining if the model has converged
        uncertainty_improvement_threshold: threshold on the uncertainty improvement of classifier during active
            learning, used for determining if the model has converged
        n_uncertainty_improvement: span of iterations to check for largest difference between uncertainties
        n_queries: maximum number of iterations to be done for the active learning session
        sampling_method: sampling method to be used for the active learning session
        verbose: sets verbosity
    """
    def __init__(self, col_names: List[str], scorer: BaseEstimator, min_nr_samples: int = 10,
                 uncertainty_threshold: float = 0.1, uncertainty_improvement_threshold: float = 0.01,
                 n_uncertainty_improvement: int = 5, n_queries: int = 9999, sampling_method=uncertainty_sampling,
                 verbose: int = 0):
        super().__init__(col_names, min_nr_samples, uncertainty_threshold, uncertainty_improvement_threshold,
                 n_uncertainty_improvement, verbose)
        self.learner = ActiveLearner(
            estimator=scorer,
            query_strategy=sampling_method
        )
        self.n_queries = n_queries


    def label_perfect_train_matches(self, identical_records: pd.DataFrame) -> None:
        """
        To prevent asking labels for the perfect matches that were created by setting `n_perfect_train_matches`, these
        are provided to the active learner upfront.

        Args:
            identical_records: Pandas dataframe containing perfect matches

        """
        identical_records['y'] = '1'
        self.learner.teach(np.array(identical_records['similarity_metrics'].values.tolist()),
                           identical_records['y'].values)
        self.train_samples = pd.concat([self.train_samples, identical_records])

    def fit(self, X: pd.DataFrame) -> 'ConfidenceLearner':
        """
        Fit ScoringLearner instance on pairs of strings
        Args:
            X: Pandas dataframe containing pairs of strings and distance metrics of paired strings
        """
        self.train_samples = pd.DataFrame([])
        query_inst_prev = None

        # automatically label all perfect train matches:
        identical_records = X[X['perfect_train_match']].copy()
        self.label_perfect_train_matches(identical_records)
        # remove identical records to avoid double labelling
        X = X.drop(identical_records.index).reset_index(drop=True)
        for _ in range(self.n_queries):
            query_idx, query_inst = self.learner.query(np.array(X['similarity_metrics'].tolist()))
            if self.learner.estimator.fitted_:
                # the uncertainty calculations need a fitted estimator
                # however it can occur that the estimator can only be fit after a couple rounds of querying
                self.calculate_uncertainty(query_inst)
                if self.verbose >= 2:
                    self.show_min_max_scores(X)
            y_new = self.get_active_learning_input(X.iloc[query_idx].iloc[0])
            if y_new == 'p':  # use previous (input is 'p')
                y_new = self.get_active_learning_input(query_inst_prev.iloc[0])
            elif y_new == 'f':  # finish labelling (input is 'f')
                break
            query_inst_prev = X.iloc[query_idx]
            if y_new != 's':  # skip case (input is 's')
                self.learner.teach(np.asarray([X.iloc[query_idx]['similarity_metrics'].iloc[0]]), np.asarray(y_new))
                train_sample_to_add = X.iloc[query_idx].copy()
                train_sample_to_add['y'] = y_new
                self.train_samples = pd.concat([self.train_samples, train_sample_to_add])
            X = X.drop(query_idx).reset_index(drop=True)
            if self.is_converged():
                print("Classifier converged, enter 'f' to stop training")
            if y_new == '1':
                self.counter_positive += 1
            elif y_new == '0':
                self.counter_negative += 1
            self.counter_total += 1
        return self

    def predict_proba(self, X: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """
        Predict probabilities on new data whether the pairs are a match or not
        Args:
            X: Pandas or Spark dataframe to predict on
        Returns: match probabilities
        """
        return self.learner.estimator.predict_proba(X)