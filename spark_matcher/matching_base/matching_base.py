# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

import warnings
from typing import Optional, List, Dict

import dill
from pyspark.sql import DataFrame, functions as F, SparkSession
import numpy as np
import pandas as pd
from thefuzz.fuzz import token_set_ratio, token_sort_ratio

from spark_matcher.activelearner.active_learner import ScoringLearner
from spark_matcher.blocker.block_learner import BlockLearner
from spark_matcher.sampler.training_sampler import HashSampler, RandomSampler
from spark_matcher.scorer.scorer import Scorer
from spark_matcher.similarity_metrics import SimilarityMetrics
from spark_matcher.blocker.blocking_rules import BlockingRule, default_blocking_rules
from spark_matcher.table_checkpointer import TableCheckpointer, ParquetCheckPointer


class MatchingBase:

    def __init__(self, spark_session: SparkSession, table_checkpointer: Optional[TableCheckpointer] = None,
                 checkpoint_dir: Optional[str] = None, col_names: Optional[List[str]] = None,
                 field_info: Optional[Dict] = None, blocking_rules: Optional[List[BlockingRule]] = None,
                 blocking_recall: float = 1.0, n_perfect_train_matches=1, n_train_samples: int = 100_000,
                 ratio_hashed_samples: float = 0.5, scorer: Optional[Scorer] = None, verbose: int = 0):
        self.spark_session = spark_session
        self.table_checkpointer = table_checkpointer
        if not self.table_checkpointer:
            if checkpoint_dir:
                self.table_checkpointer = ParquetCheckPointer(self.spark_session, checkpoint_dir,
                                                              "checkpoint_deduplicator")
            else:
                warnings.warn(
                    'Either `table_checkpointer` or `checkpoint_dir` should be provided. This instance can only be used'
                    ' when loading a previously saved instance.')

        if col_names:
            self.col_names = col_names
            self.field_info = {col_name: [token_set_ratio, token_sort_ratio] for col_name in
                               self.col_names}
        elif field_info:
            self.col_names = list(field_info.keys())
            self.field_info = field_info
        else:
            warnings.warn(
                'Either `col_names` or `field_info` should be provided. This instance can only be used when loading a '
                'previously saved instance.')
            self.col_names = ['']  # needed for instantiating ScoringLearner

        self.n_train_samples = n_train_samples
        self.ratio_hashed_samples = ratio_hashed_samples
        self.n_perfect_train_matches = n_perfect_train_matches
        self.verbose = verbose

        if not scorer:
            scorer = Scorer(self.spark_session)
        self.scoring_learner = ScoringLearner(self.col_names, scorer, verbose=self.verbose)

        self.blocking_rules = blocking_rules
        if not self.blocking_rules:
            self.blocking_rules = [blocking_rule(col) for col in self.col_names for blocking_rule in
                                   default_blocking_rules]
        self.blocker = BlockLearner(blocking_rules=self.blocking_rules, recall=blocking_recall,
                                    table_checkpointer=self.table_checkpointer, verbose=self.verbose)

    def save(self, path: str) -> None:
        """
        Save the current instance to a pickle file.

        Args:
            path: Path and file name of pickle file

        """
        to_save = self.__dict__

        # the spark sessions need to be removed as they cannot be saved and re-used later
        to_save['spark_session'] = None
        setattr(to_save['scoring_learner'].learner.estimator, 'spark_session', None)
        setattr(to_save['table_checkpointer'], 'spark_session', None)

        with open(path, 'wb') as f:
            dill.dump(to_save, f)

    def load(self, path: str) -> None:
        """
        Load a previously trained and saved Matcher instance.

        Args:
            path: Path and file name of pickle file

        """
        with open(path, 'rb') as f:
            loaded_obj = dill.load(f)

        # the spark session that was removed before saving needs to be filled with the spark session of this instance
        loaded_obj['spark_session'] = self.spark_session
        setattr(loaded_obj['scoring_learner'].learner.estimator, 'spark_session', self.spark_session)
        setattr(loaded_obj['table_checkpointer'], 'spark_session', self.spark_session)

        self.__dict__.update(loaded_obj)

    def _simplify_dataframe_for_matching(self, sdf: DataFrame) -> DataFrame:
        """
        Select only the columns used for matching and drop duplicates.
        """
        return sdf.select(*self.col_names).drop_duplicates()

    def _create_train_pairs_table(self, sdf_1: DataFrame, sdf_2: Optional[DataFrame] = None) -> DataFrame:
        """
        Create pairs that are used for training. Based on the given 'ratio_hashed_samples', part of the sample
        pairs are generated using MinHashLSH technique to create pairs that are more likely to be a match. Rest is
        sampled using random selection.

        Args:
            sdf_1: Spark dataframe containing the first table to with the input should be matched
            sdf_2: Optional: Spark dataframe containing the second table that should be matched to the first table

        Returns:
            Spark dataframe with sampled pairs that should be compared during training
        """
        n_hashed_samples = int(self.n_train_samples * self.ratio_hashed_samples)

        h_sampler = HashSampler(self.table_checkpointer, self.col_names, n_hashed_samples)
        hashed_pairs_table = self.table_checkpointer.checkpoint_table(h_sampler.create_pairs_table(sdf_1, sdf_2),
                                                                      checkpoint_name='minhash_pairs_table')

        # creating additional random samples to ensure that the exact number of n_train_samples
        # is obtained after dropping duplicated records
        n_random_samples = int(1.5 * (self.n_train_samples - hashed_pairs_table.count()))

        r_sampler = RandomSampler(self.table_checkpointer, self.col_names, n_random_samples)
        random_pairs_table = self.table_checkpointer.checkpoint_table(r_sampler.create_pairs_table(sdf_1, sdf_2),
                                                                      checkpoint_name='random_pairs_table')

        pairs_table = (
            random_pairs_table
            .unionByName(hashed_pairs_table)
            .withColumn('perfect_train_match', F.lit(False))
            .drop_duplicates()
            .limit(self.n_train_samples)
        )

        # add some perfect matches to assure there are two labels in the train data and the classifier can be trained
        perfect_matches = (pairs_table.withColumn('perfect_train_match', F.lit(True))
                           .limit(self.n_perfect_train_matches))
        for col in self.col_names:
            perfect_matches = perfect_matches.withColumn(col + "_1", F.col(col + "_2"))

        pairs_table = (perfect_matches.unionByName(pairs_table
                                                   .limit(self.n_train_samples - self.n_perfect_train_matches)))

        return pairs_table

    def _calculate_metrics(self, pairs_table: DataFrame) -> DataFrame:
        """
        Method to apply similarity metrics to pairs table.

        Args:
            pairs_table: Spark dataframe containing pairs table

        Returns:
            Spark dataframe with pairs table and newly created `similarity_metrics` column

        """
        similarity_metrics = SimilarityMetrics(self.field_info)
        return similarity_metrics.transform(pairs_table)

    def _create_blocklearning_input(self, metrics_table: pd.DataFrame, threshold: int = 0.5) -> DataFrame:
        """
        Method to collect data used for block learning. This data consists of the manually labelled pairs with label 1
        and the train pairs with a score above the `threshold`

        Args:
            metrics_table: Pandas dataframe containing the similarity metrics
            threshold: scoring threshold

        Returns:
            Spark dataframe with data for block learning

        """
        metrics_table['score'] = self.scoring_learner.predict_proba(np.array(metrics_table['similarity_metrics'].tolist()))[:, 1]
        metrics_table.loc[metrics_table['score'] > threshold, 'label'] = '1'

        # get labelled positives from activelearner
        positive_train_labels = (self.scoring_learner.train_samples[self.scoring_learner.train_samples['y'] == '1']
                                 .rename(columns={'y': 'label'}))
        metrics_table = pd.concat([metrics_table, positive_train_labels]).drop_duplicates(
            subset=[col + "_1" for col in self.col_names] + [col + "_2" for col in self.col_names], keep='last')

        metrics_table = metrics_table[metrics_table.label == '1']

        metrics_table['row_id'] = np.arange(len(metrics_table))
        return self.spark_session.createDataFrame(metrics_table)

    def _add_suffix_to_col_names(self, sdf: DataFrame, suffix: int):
        """
        This method adds a suffix to the columns that are used in the algorithm.
        This is done in order to do a join (with two dataframes with the same schema) to create the pairs table.
        """
        for col in self.col_names:
            sdf = sdf.withColumnRenamed(col, f"{col}_{suffix}")
        return sdf

    def fit(self, sdf_1: DataFrame, sdf_2: Optional[DataFrame] = None) -> 'MatchingBase':
        """
        Fit the MatchingBase instance on the two dataframes `sdf_1` and `sdf_2` using active learning. You will be
        prompted to enter whether the presented pairs are a match or not. Note that `sdf_2` is an optional argument.
        `sdf_2` is used for Matcher (i.e. matching one table to another). In the case of Deduplication, only providing
        `sdf_1` is sufficient, in that case `sdf_1` will be deduplicated.

        Args:
            sdf_1: Spark dataframe
            sdf_2: Optional: Spark dataframe

        Returns:
            Fitted MatchingBase instance

        """
        sdf_1 = self._simplify_dataframe_for_matching(sdf_1)
        if sdf_2:
            sdf_2 = self._simplify_dataframe_for_matching(sdf_2)

        pairs_table = self._create_train_pairs_table(sdf_1, sdf_2)
        metrics_table = self._calculate_metrics(pairs_table)
        metrics_table_pdf = metrics_table.toPandas()
        self.scoring_learner.fit(metrics_table_pdf)
        block_learning_input = self._create_blocklearning_input(metrics_table_pdf)
        self.blocker.fit(block_learning_input)
        self.fitted_ = True
        return self
