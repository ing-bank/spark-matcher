# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import Optional, List, Dict

from pyspark.sql import DataFrame, functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window
from sklearn.exceptions import NotFittedError

from spark_matcher.blocker.blocking_rules import BlockingRule
from spark_matcher.matching_base.matching_base import MatchingBase
from spark_matcher.scorer.scorer import Scorer
from spark_matcher.table_checkpointer import TableCheckpointer


class Matcher(MatchingBase):
    """
    Matcher class to apply record linkage. Provide either the column names `col_names` using the default string
    similarity metrics or explicitly define the string similarity metrics in a dict `field_info` as in the example
    below. If `blocking_rules` is left empty, default blocking rules are used. Otherwise provide blocking rules as
    a list containing `BlockingRule` instances (see example below). The number of perfect matches used during
    training is set by `n_perfect_train_matches`.

    E.g.:

    from spark_matcher.blocker.blocking_rules import FirstNChars

    myMatcher = Matcher(spark_session, field_info={'name':[metric_function_1, metric_function_2],
                                                   'address:[metric_function_1, metric_function_3]},
                                                   blocking_rules=[FirstNChars('name', 3)])

    Args:
        spark_session: Spark session
        col_names: list of column names to use for matching
        field_info: dict of column names as keys and lists of string similarity metrics as values
        blocking_rules: list of `BlockingRule` instances
        n_train_samples: nr of pair samples to be created for training
        ratio_hashed_samples: ratio of hashed samples to be created for training, rest is sampled randomly
        n_perfect_train_matches: nr of perfect matches used for training
        scorer: a Scorer object used for scoring pairs
        verbose: sets verbosity
    """
    def __init__(self, spark_session: SparkSession, table_checkpointer: Optional[TableCheckpointer]=None,
                 checkpoint_dir: Optional[str]=None, col_names: Optional[List[str]] = None,
                 field_info: Optional[Dict] = None, blocking_rules: Optional[List[BlockingRule]] = None,
                 blocking_recall: float = 1.0, n_perfect_train_matches=1, n_train_samples: int = 100_000,
                 ratio_hashed_samples: float = 0.5, scorer: Optional[Scorer] = None, verbose: int = 0):
        super().__init__(spark_session, table_checkpointer, checkpoint_dir, col_names, field_info, blocking_rules,
                         blocking_recall, n_perfect_train_matches, n_train_samples, ratio_hashed_samples, scorer,
                         verbose)
        self.fitted_ = False

    def _create_predict_pairs_table(self, sdf_1_blocked: DataFrame, sdf_2_blocked: DataFrame) -> DataFrame:
        """
        Method to create pairs within blocks as provided by both input Spark dataframes in the column `block_key`.
        Note that an inner join is performed to create pairs: entries in `sdf_1_blocked` and `sdf_2_blocked`
        that don't share a `block_key` value in the other table will not appear in the resulting pairs table.

        Args:
            sdf_1_blocked: Spark dataframe containing all columns used for matching and the `block_key`
            sdf_2_blocked: Spark dataframe containing all columns used for matching and the `block_key`

        Returns:
            Spark dataframe containing all pairs to score

        """
        sdf_1_blocked = self._add_suffix_to_col_names(sdf_1_blocked, 1)
        sdf_2_blocked = self._add_suffix_to_col_names(sdf_2_blocked, 2)

        predict_pairs_table = (sdf_1_blocked.join(sdf_2_blocked, on='block_key', how='inner')
                               .drop_duplicates(subset=[col + "_1" for col in self.col_names] +
                                                       [col + "_2" for col in self.col_names])
                               )
        return predict_pairs_table

    def predict(self, sdf_1, sdf_2, threshold=0.5, top_n=None):
        """
        Method to predict on data used for training or new data.

        Args:
            sdf_1: first table
            sdf_2: second table
            threshold: probability threshold
            top_n: only return best `top_n` matches above threshold

        Returns:
            Spark dataframe with the matching result

        """
        if not self.fitted_:
            raise NotFittedError('The Matcher instance is not fitted yet. Call `fit` and train the instance.')

        sdf_1, sdf_2 = self._simplify_dataframe_for_matching(sdf_1), self._simplify_dataframe_for_matching(sdf_2)
        sdf_1_blocks, sdf_2_blocks = self.blocker.transform(sdf_1, sdf_2)
        pairs_table = self._create_predict_pairs_table(sdf_1_blocks, sdf_2_blocks)
        metrics_table = self._calculate_metrics(pairs_table)
        scores_table = (
            metrics_table
            .withColumn('score', self.scoring_learner.predict_proba(metrics_table['similarity_metrics']))
        )
        scores_table_filtered = (scores_table.filter(F.col('score') >= threshold)
                                 .drop('block_key', 'similarity_metrics'))

        if top_n:
            # we add additional columns to order by to remove ties and return exactly n items
            window = (Window.partitionBy(*[x + "_1" for x in self.col_names])
                      .orderBy(F.desc('score'), *[F.asc(col) for col in [x + "_2" for x in self.col_names]]))
            result = (scores_table_filtered.withColumn('rank', F.rank().over(window))
                      .filter(F.col('rank') <= top_n)
                      .drop('rank'))
        else:
            result = scores_table_filtered

        return result
