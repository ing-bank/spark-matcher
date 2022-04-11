# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from pyspark.sql import DataFrame
from sklearn.base import BaseEstimator


class ScoringLearner:
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
        self.col_names = col_names
        self.learner = ActiveLearner(
            estimator=scorer,
            query_strategy=sampling_method
        )
        self.counter_total = 0
        self.counter_positive = 0
        self.counter_negative = 0
        self.min_nr_samples = min_nr_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_improvement_threshold = uncertainty_improvement_threshold
        self.n_uncertainty_improvement = n_uncertainty_improvement
        self.uncertainties = []
        self.n_queries = n_queries
        self.verbose = verbose

    def _input_assert(self, message: str, choices: List[str]) -> str:
        """
        Adds functionality to the python function `input` to limit the choices that can be returned
        Args:
            message: message to user
            choices: list containing possible choices that can be returned
        Returns:
            input returned by user
        """
        output = input(message).lower()
        if output not in choices:
            print(f"Wrong input! Your input should be one of the following: {', '.join(choices)}")
            return self._input_assert(message, choices)
        else:
            return output

    def _get_uncertainty_improvement(self) -> Optional[float]:
        """
        Calculates the uncertainty differences during active learning. The largest difference over the `last_n`
        iterations is returned. The aim of this function is to suggest early stopping of active learning.

        Returns: largest uncertainty update in `last_n` iterations

        """
        uncertainties = np.asarray(self.uncertainties)
        abs_differences = abs(uncertainties[1:] - uncertainties[:-1])
        return max(abs_differences[-self.n_uncertainty_improvement:])

    def _is_converged(self) -> bool:
        """
        Checks whether the model is converged by comparing the last uncertainty value with the `uncertainty_threshold`
        and comparing the `last_n` uncertainty improvements with the `uncertainty_improvement_threshold`. These checks
        are only performed if at least `min_nr_samples` are labelled.

        Returns:
            boolean indicating whether the model is converged

        """
        if (self.counter_total >= self.min_nr_samples) and (
                len(self.uncertainties) >= self.n_uncertainty_improvement + 1):
            uncertainty_improvement = self._get_uncertainty_improvement()
            if (self.uncertainties[-1] <= self.uncertainty_threshold) or (
                    uncertainty_improvement <= self.uncertainty_improvement_threshold):
                return True
        else:
            return False

    def _get_active_learning_input(self, query_inst: pd.DataFrame) -> np.ndarray:
        """
        Obtain user input for a query during active learning.
        Args:
            query_inst: query as provided by the ActiveLearner instance
        Returns: label of user input '1' or '0' as yes or no
                    'p' to go to previous
                    'f' to finish
                    's' to skip the query
        """
        print(f'\nNr. {self.counter_total + 1} ({self.counter_positive}+/{self.counter_negative}-)')
        print("Is this a match? (y)es, (n)o, (p)revious, (s)kip, (f)inish")
        print('')
        for element in [1, 2]:
            for col_name in self.col_names:
                print(f'{col_name}_{element}' + ': ' + query_inst[f'{col_name}_{element}'].iloc[0])
            print('')
        user_input = self._input_assert("", ['y', 'n', 'p', 'f', 's'])
        # replace 'y' and 'n' with '1' and '0' to make them valid y labels
        user_input = user_input.replace('y', '1').replace('n', '0')

        y_new = np.array([user_input])
        return y_new

    def _calculate_uncertainty(self, x) -> None:
        # take the maximum probability of the predicted classes as proxy of the confidence of the classifier
        confidence = self.predict_proba(x).max(axis=1)[0]
        if self.verbose:
            print('uncertainty:', 1 - confidence)
        self.uncertainties.append(1 - confidence)

    def _show_min_max_scores(self, X: pd.DataFrame) -> None:
        """
        Prints the lowest and the highest logistic regression scores on train data during active learning.

        Args:
            X: Pandas dataframe containing train data that is available for labelling duringg active learning
        """
        X_all = pd.concat((X, self.train_samples))
        pred_max = self.learner.predict_proba(np.array(X_all['similarity_metrics'].tolist())).max(axis=0)
        print(f'lowest score: {1 - pred_max[0]:.3f}')
        print(f'highest score: {pred_max[1]:.3f}')

    def _label_perfect_train_matches(self, identical_records: pd.DataFrame) -> None:
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

    def fit(self, X: pd.DataFrame) -> 'ScoringLearner':
        """
        Fit ScoringLearner instance on pairs of strings
        Args:
            X: Pandas dataframe containing pairs of strings and distance metrics of paired strings
        """
        self.train_samples = pd.DataFrame([])
        query_inst_prev = None

        # automatically label all perfect train matches:
        identical_records = X[X['perfect_train_match']].copy()
        self._label_perfect_train_matches(identical_records)
        X = X.drop(identical_records.index).reset_index(drop=True)  # remove identical records to avoid double labelling

        for i in range(self.n_queries):
            query_idx, query_inst = self.learner.query(np.array(X['similarity_metrics'].tolist()))

            if self.learner.estimator.fitted_:
                # the uncertainty calculations need a fitted estimator
                # however it can occur that the estimator can only be fit after a couple rounds of querying
                self._calculate_uncertainty(query_inst)
                if self.verbose >= 2:
                    self._show_min_max_scores(X)

            y_new = self._get_active_learning_input(X.iloc[query_idx])
            if y_new == 'p':  # use previous (input is 'p')
                y_new = self._get_active_learning_input(query_inst_prev)
            elif y_new == 'f':  # finish labelling (input is 'f')
                break
            query_inst_prev = X.iloc[query_idx]
            if y_new != 's':  # skip case (input is 's')
                self.learner.teach(np.asarray([X.iloc[query_idx]['similarity_metrics'].iloc[0]]), np.asarray(y_new))
                train_sample_to_add = X.iloc[query_idx].copy()
                train_sample_to_add['y'] = y_new
                self.train_samples = pd.concat([self.train_samples, train_sample_to_add])

            X = X.drop(query_idx).reset_index(drop=True)

            if self._is_converged():
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
