from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class ActiveLearnerBase(ABC):
    """
    A base class for active learning approaches.
    Attributes:
        col_names: column names used for matching
        scorer: the scorer to be used in the active learning loop
        min_nr_samples: minimum number of responses required before classifier convergence is tested
        uncertainty_threshold: threshold on the uncertainty of the classifier during active learning,
            used for determining if the model has converged
        uncertainty_improvement_threshold: threshold on the uncertainty improvement of classifier during active
            learning, used for determining if the model has converged
        n_uncertainty_improvement: span of iterations to check for largest difference between uncertainties
        verbose: sets verbosity
    """
    def __init__(self, col_names: List[str], min_nr_samples: int = 10,
                 uncertainty_threshold: float = 0.1, uncertainty_improvement_threshold: float = 0.01,
                 n_uncertainty_improvement: int = 5, verbose: int = 0):
        self.col_names = col_names
        self.counter_total = 0
        self.counter_positive = 0
        self.counter_negative = 0
        self.min_nr_samples = min_nr_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_improvement_threshold = uncertainty_improvement_threshold
        self.n_uncertainty_improvement = n_uncertainty_improvement
        self.uncertainties = []
        self.train_samples = pd.DataFrame([])
        self.verbose = verbose

    def input_assert(self, message: str, choices: List[str]) -> str:
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
            return self.input_assert(message, choices)
        return output

    def get_uncertainty_improvement(self) -> Optional[float]:
        """
        Calculates the uncertainty differences during active learning. The largest difference over the `last_n`
        iterations is returned. The aim of this function is to suggest early stopping of active learning.

        Returns: largest uncertainty update in `last_n` iterations

        """
        uncertainties = np.asarray(self.uncertainties)
        abs_differences = abs(uncertainties[1:] - uncertainties[:-1])
        return max(abs_differences[-self.n_uncertainty_improvement:])

    def is_converged(self) -> bool:
        """
        Checks whether the model is converged by comparing the last uncertainty value with the `uncertainty_threshold`
        and comparing the `last_n` uncertainty improvements with the `uncertainty_improvement_threshold`. These checks
        are only performed if at least `min_nr_samples` are labelled.

        Returns:
            boolean indicating whether the model is converged

        """
        if (self.counter_total >= self.min_nr_samples) and (
                len(self.uncertainties) >= self.n_uncertainty_improvement + 1):
            uncertainty_improvement = self.get_uncertainty_improvement()
            if (self.uncertainties[-1] <= self.uncertainty_threshold) or (
                    uncertainty_improvement <= self.uncertainty_improvement_threshold):
                return True
        else:
            return False

    def get_active_learning_input(self, x: pd.Series) -> np.ndarray:
        """
        Obtain user input for a query during active learning.
        Args:
            x: query as provided by the ActiveLearner instance
        Returns: label of user input '1' or '0' as yes or no
                    'p' to go to previous
                    'f' to finish
                    's' to skip the query
        """
        print(f'\nNr. {self.counter_total + 1} ({self.counter_positive}+/{self.counter_negative}-)')
        print("Is this a match? (y)es, (n)o, (p)revious, (s)kip, (f)inish")
        print(' ')
        for element in [1, 2]:
            for col_name in self.col_names:
                print(f'{col_name}_{element}' + ': ' + x[f'{col_name}_{element}'])
            print('')
        user_input = self.input_assert("", choices = ['y', 'n', 'p', 'f', 's'])
        # replace 'y' and 'n' with '1' and '0' to make them valid y labels
        user_input = user_input.replace('y', '1').replace('n', '0')
        y_new = np.array([user_input])
        return y_new
    
    def _batch_uncertainty(self, x: np.ndarray) -> None:
        """
        This function calculates average of uncertainty with lower/upper confidence level for a given batch of data
        """
        classwise_certainty = self.predict_proba(x)
        uncertainty = 1 - np.max(classwise_certainty, axis=1)
        idx = np.arange(uncertainty.shape[0])
        rng = np.random.RandomState(seed=1234)
        samples_uncertainty = []
        for _ in range(200):
            pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
            uncertainty_boot  = np.mean(uncertainty[pred_idx])
            samples_uncertainty.append(uncertainty_boot)
        bootstrap_mean = np.mean(samples_uncertainty)
        ci_lower = np.percentile(samples_uncertainty, 2.5)
        ci_upper = np.percentile(samples_uncertainty, 97.5)
        if self.verbose:
            print(f"""The average uncertainty of model for given batch is {round(bootstrap_mean, ndigits=3)}
             with lower and upper confidence of [{round(ci_lower, ndigits=3)}, {round(ci_upper, ndigits=3)}].""")
        self.uncertainties.append(round(bootstrap_mean, ndigits=3))

    def calculate_uncertainty(self, x: np.ndarray) -> None:
        # take the maximum probability of the predicted classes as proxy of the confidence of the classifier
        if x.shape[0] > 1 : 
            self._batch_uncertainty(x)
        else:
            confidence = self.predict_proba(x).max(axis=1)[0]
            if self.verbose:
                print('The uncertainty of selected sample is:', round(1 - confidence, ndigits=3))
            self.uncertainties.append(round(1 - confidence, ndigits=3))

    def show_min_max_scores(self, X: pd.DataFrame) -> None:
        """
        Prints the lowest and the highest logistic regression scores on train data during active learning.

        Args:
            X: Pandas dataframe containing train data that is available for labelling duringg active learning
        """
        X_all = pd.concat((X, self.train_samples))
        pred_max = self.predict_proba(np.array(X_all['similarity_metrics'].tolist())).max(axis=0)
        print(f"""The lowest and highest score of model for the entire dataset are :
                [{1 - pred_max[0]:.3f},  {pred_max[1]:.3f}]""")

    @abstractmethod
    def label_perfect_train_matches(self, *args, **kwargs) -> None:
        """
        To prevent asking labels for the perfect matches, this function provide them to the active learner upfront.
        """
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        fit the active learner instance on data
        """
        pass

    @abstractmethod
    def predict_proba(self, *args, **kwargs):
        """
        predict results using trained model
        """
        pass