from typing import List, Union
from cardinal.zhdanov2019 import TwoStepKMeansSampler
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from spark_matcher.activelearner.active_learner_base import ActiveLearnerBase


class TwoStepKMeansSamplerExtended(TwoStepKMeansSampler):
    """Extends TwoStepKMeansSampler class to include uncertainty score
    """
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):
        super().__init__(beta, classifier, batch_size, assume_fitted, verbose, **kmeans_args)
        self.estimator = self.sampler_list[0].classifier_


class DiverseMiniBatchLearner(ActiveLearnerBase):
    """
    Class to train a string matching model using active learning.
    Attributes:
        col_names: column names used for matching
        scorer: the scorer to be used in the active learning loop
        batch_size: the size of batch for labeling
        min_nr_batch: minimum number of batch required before classifier convergence is tested
        uncertainty_threshold: threshold on the uncertainty of the classifier during active learning,
            used for determining if the model has converged
        uncertainty_improvement_threshold: threshold on the uncertainty improvement of classifier during active
            learning, used for determining if the model has converged
        n_uncertainty_improvement: span of iterations to check for largest difference between uncertainties
        n_queries: maximum number of iterations to be done for the active learning session
        verbose: sets verbosity
    """
    def __init__(self, col_names: List[str], scorer: BaseEstimator, beta: int = 5, 
                batch_size: int = 5, min_nr_batch: int = 2,
                uncertainty_threshold: float = 0.2, uncertainty_improvement_threshold: float = 0.01,
                n_uncertainty_improvement: int = 1, verbose: int = 0
                ):
        min_nr_samples = min_nr_batch * batch_size
        super().__init__(col_names, min_nr_samples, uncertainty_threshold, uncertainty_improvement_threshold,
                        n_uncertainty_improvement, verbose
                        )
        self.batch_size = batch_size
        self.learner = TwoStepKMeansSamplerExtended(beta,
                                                    classifier=scorer,
                                                    batch_size=batch_size
                                                    )
    
    def _process_input_batch(self, query_inst: pd.DataFrame) -> dict:
        """
        Process user input for a give sample of data
        Args:
            query_inst (pd.DataFrame): sample of data to be labelled by user

        Returns:
            dict: label of user input as a dict, for example {'index number': '1'}
        """
        # checking after each batch if the model is converged
        if self.is_converged():
            print("Classifier converged, enter 'f' to stop training")
            
        y_new = {}
        for index, row in query_inst.iterrows():
            user_input = self.get_active_learning_input(row)[0]
            
            if user_input == 'p':
                if y_new:
                    prev_index = list(y_new.keys())[-1]
                    user_input_prev = self.get_active_learning_input(query_inst.loc[prev_index])[0]
                    y_new[prev_index] = user_input_prev
                else:
                    print('Model is already trained on previous batch')
                # asking again the current sample to be labeled
                user_input = self.get_active_learning_input(row)[0]
            # set up a counter
            if user_input == '1':
                self.counter_positive += 1
            elif user_input == '0':
                self.counter_negative += 1
            elif user_input == 'f':
                y_new[index] = user_input
                return y_new
            self.counter_total += 1
            y_new[index] = user_input
        return y_new
        
    def label_perfect_train_matches(self, identical_records: pd.DataFrame) -> None:
        """
        To prevent asking labels for the perfect matches that were created by setting `n_perfect_train_matches`, these
        are provided to the active learner upfront.

        Args:
            identical_records: Pandas dataframe containing perfect matches

        """
        identical_records['y'] = '1'
        # adding one negative sample to train the model on both classes 
        n_feature = len(identical_records['similarity_metrics'].values.tolist()[0])
        x_perfect = np.append(np.array(identical_records['similarity_metrics'].values.tolist())
                            , np.zeros((1,n_feature)), axis=0)
        y_perfect = np.append(identical_records['y'].values, np.array(['0']), axis=0)
        
        # fitting the learner
        self.learner.fit(x_perfect, y_perfect)
        self.train_samples = pd.concat([self.train_samples, identical_records])

    def fit(self, X: pd.DataFrame) -> 'DiverseMiniBatchLearner':
        """
        Fit ScoringLearner instance on pairs of strings
        Args:
            X: Pandas dataframe containing pairs of strings and distance metrics of paired strings
        """
        # automatically label all perfect train matches:
        identical_records = X[X['perfect_train_match']].copy()
        self.label_perfect_train_matches(identical_records)
        # remove identical records to avoid double labelling
        X = X.drop(identical_records.index).reset_index(drop=True)
        # number of iterations over batches
        n_iter = int(X.shape[0] // self.batch_size)
        for i in range(n_iter):
            # selecting first batch from the pool
            query_index = self.learner.select_samples(np.array(X['similarity_metrics'].tolist()))
            # before labeling, insights about what is the current uncertanty and min(max) of prediction
            if self.verbose >= 2:
                self.calculate_uncertainty(np.array(X.iloc[query_index]['similarity_metrics'].tolist()))
                self.show_min_max_scores(X)
            # labeling the selected batch from pool
            y_new = self._process_input_batch(X.iloc[query_index])
            # if users decides to finish labeling
            if 'f' in y_new.values():
                break
            # processing labelled samples and removing 's' ones or 'p'
            removed_skipped_feedback = {key:value for key, value in y_new.items()
                                        if (value !='s') or (value !='p')}
            train_sample_to_add = X.iloc[[*removed_skipped_feedback]].copy()
            train_sample_to_add['y'] = list(removed_skipped_feedback.values())
            self.train_samples = pd.concat([self.train_samples, train_sample_to_add])
            # update the pool by removing already labeled batch
            X = X.drop([*removed_skipped_feedback]).reset_index(drop=True)
            # training the model with new labeled data
            self.learner.fit(
                        np.array(self.train_samples['similarity_metrics'].values.tolist()),
                                     self.train_samples['y'].values)
            if self.verbose >= 2:
                print(f"The batch number {i+1} for labeling is done.")  
        return self

    def predict_proba(self, X: Union[pd.DataFrame, pd.DataFrame]) -> Union[pd.DataFrame, pd.DataFrame]:
        """
        Predict probabilities on new data whether the pairs are a match or not
        Args:
            X: Pandas or Spark dataframe to predict on
        Returns: match probabilities
        """
        return self.learner.sampler_list[0].classifier_.predict_proba(X)