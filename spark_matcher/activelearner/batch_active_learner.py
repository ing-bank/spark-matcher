import random
from spark_matcher.scorer.scorer import Scorer
from cardinal.zhdanov2019 import TwoStepKMeansSampler
from sklearn.base import BaseEstimator
import numpy as np
from typing import List, Optional, Union
import pandas as pd


        
class TwoStepKMeansSamplerExtended(TwoStepKMeansSampler):
    """Extends TwoStepKMeansSampler class to include confidence score
    """

    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):
        super().__init__(beta, classifier, batch_size, assume_fitted, verbose, **kmeans_args)
    
    def _get_probability_classes(self, X: np.ndarray) -> np.ndarray:
        """Returns classifier.predict_proba(X)
        Args:
            classifier: The classifier for which probabilities are to be queried.
            X: Samples to classify.
        Returns:
            The probability of each class for each sample.
        """
        if self.sampler_list[0].classifier_ == 'precomputed':
            return X
        classwise_uncertainty = self.sampler_list[0].classifier_.predict_proba(X)
        return classwise_uncertainty
    
    def confidence_score(self, X: np.ndarray) -> np.ndarray:
        """Measure the confidence score of a model for a set of samples.
        Args:
            classifier: The classifier for which the labels are to be queried.
            X: The pool of samples to query from.
        Returns:
            The confidence score for each sample.
        """
        classwise_uncertainty = self._get_probability_classes(X)
        uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
        return uncertainty


class DiverseMiniBatchLearner:
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
    def __init__(self, col_names: List[str], scorer: BaseEstimator, beta: int = 5, batch_size: int = 10, min_nr_samples: int = 10,
                 uncertainty_threshold: float = 0.1, uncertainty_improvement_threshold: float = 0.01,
                 n_uncertainty_improvement: int = 5, n_queries: int = 9999, sampling_method=uncertainty_sampling,
                 verbose: int = 0):
        self.batch_size = batch_size
        self.col_names = col_names
        self.learner = TwoStepKMeansSamplerExtended(beta,
                                            classifier=scorer,
                                            batch_size=batch_size
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
    
    def _get_active_learning_input_batch(self, query_inst: pd.DataFrame) -> dict:
        """
        Obtain user input for a query during active learning.
        Args:
            query_inst: query as provided by the ActiveLearner instance
        Returns: label of user input '1' or '0' as yes or no
                    'p' to go to previous
                    'f' to finish
                    's' to skip the query
        """
        # checking after each batch if the model is converged
        if self._is_converged():
            print("Classifier converged, enter 'f' to stop training")
            
        y_new = {}
        for index, row in query_inst.iterrows():
            print(f'\nNr. {self.counter_total + 1} ({self.counter_positive}+/{self.counter_negative}-)')
            print("Is this a match? (y)es, (n)o , (s)kip, (f)inish")
            print('')
            for element in [1, 2]:
                for col_name in self.col_names:
                    print(f'{col_name}_{element}' + ': ' + row[f'{col_name}_{element}'])
                print('')
            user_input = self._input_assert("", ['y', 'n', 's', 'f'])
            # replace 'y' and 'n' with '1' and '0' to make them valid y labels
            user_input = user_input.replace('y', '1').replace('n', '0')
            # set up a counter
            if user_input == '1':
                self.counter_positive += 1
            elif user_input == '0':
                self.counter_negative += 1
            elif user_input == 'f':
                y_new[index] = user_input
                return y_new
            self.counter_total += 1
            y_new[index]= user_input
        return y_new

    def _calculate_uncertainty(self, x) -> None:
        # take the maximum probability of the predicted classes as proxy of the confidence of the classifier
        confidence = self.learner.confidence_score(x)
        if self.verbose:
            print(f'average uncertainty: {round(np.mean(1 - confidence), ndigits=3)}')
        self.uncertainties.append(round(np.mean(1 - confidence), ndigits=3))

    def _show_min_max_scores(self, X: pd.DataFrame) -> None:
        """
        Prints the lowest and the highest logistic regression scores on train data during active learning.

        Args:
            X: Pandas dataframe containing train data that is available for labelling duringg active learning
        """
        X_all = pd.concat((X, self.train_samples))
        pred_max = self.learner.sampler_list[0].classifier_.predict_proba(np.array(X_all['similarity_metrics'].tolist())).max(axis=0)
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
        # adding one negative sample to train the model on both classes 
        n_feature = len(identical_records['similarity_metrics'].values.tolist()[0])
        x_perfect = np.append(np.array(identical_records['similarity_metrics'].values.tolist()), np.zeros((1,n_feature)), axis=0)
        y_prefect = np.append(identical_records['y'].values, np.array(['0']), axis=0)
        
        # fitting the learner
        self.learner.fit(x_perfect, y_prefect)
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
        
        # number of iterations over batches
        N_ITER = X.shape[0] // self.batch_size
        
        for i in range(N_ITER):
            
            # selecting first batch from the pool
            query_index = self.learner.select_samples(np.array(X['similarity_metrics'].tolist()))
            
            # before labeling, insights about what is the current uncertanty
            self._calculate_uncertainty(np.array(X.iloc[query_index]['similarity_metrics'].tolist()))
            if self.verbose >= 2:
                self._show_min_max_scores(X.iloc[query_index])
            
            # labeling the selected batch from pool
            y_new = self._get_active_learning_input_batch(X.iloc[query_index])
               
            
            # processing labelled samples and removing 's' ones
            query_inst_prev = X.iloc[[*y_new]]
            removed_skipped_feedabck = {key:value for key, value in y_new.items() if value !='s'}
            train_sample_to_add = X.iloc[[*removed_skipped_feedabck]].copy()
            train_sample_to_add['y'] = list(removed_skipped_feedabck.values())
            self.train_samples = pd.concat([self.train_samples, train_sample_to_add])
                
            # update the pool by removing already labeled batch
            X = X.drop([*removed_skipped_feedabck]).reset_index(drop=True)
            
            # training the model with new labeled data, there is one edge case though if there is not a single negative case it fails
            self.learner.fit(
                        np.array(self.train_samples['similarity_metrics'].values.tolist()),
                                     self.train_samples['y'].values)
            print(f"The {i+1} batch of samples for labeling is done!")  
        return self