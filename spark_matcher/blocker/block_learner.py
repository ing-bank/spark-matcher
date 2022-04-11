# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import List, Tuple, Optional, Union

from pyspark.sql import DataFrame, functions as F

from spark_matcher.blocker.blocking_rules import BlockingRule
from spark_matcher.table_checkpointer import TableCheckpointer


class BlockLearner:
    """
    Class to learn blocking rules from training data.

    Attributes:
        blocking_rules: list of `BlockingRule` objects that are taken into account during block learning
        recall: the minimum required percentage of training pairs that are covered by the learned blocking rules
        verbose: set verbosity
    """
    def __init__(self, blocking_rules: List[BlockingRule], recall: float,
                 table_checkpointer: Optional[TableCheckpointer] = None, verbose=0):
        self.table_checkpointer = table_checkpointer
        self.blocking_rules = blocking_rules
        self.recall = recall
        self.cover_blocking_rules = None
        self.fitted = False
        self.full_set = None
        self.full_set_size = None
        self.verbose = verbose

    def _greedy_set_coverage(self):
        """
        This method solves the `set cover problem` with a greedy algorithm. It identifies a subset of blocking-rules
        that cover `recall` * |full_set| (|full_set| stands for cardinality of the full_set)percent of all the elements
        in the original (full) set.
        """
        # sort the blocking rules to start the algorithm with the ones that have most coverage:
        _sorted_blocking_rules = sorted(self.blocking_rules, key=lambda bl: bl.training_coverage_size, reverse=True)

        self.cover_blocking_rules = [_sorted_blocking_rules.pop(0)]
        self.cover_set = self.cover_blocking_rules[0].training_coverage

        for blocking_rule in _sorted_blocking_rules:
            # check if the required recall is already reached:
            if len(self.cover_set) >= int(self.recall * self.full_set_size):
                break
            # check if subset is dominated by the cover_set:
            if blocking_rule.training_coverage.issubset(self.cover_set):
                continue
            self.cover_set = self.cover_set.union(blocking_rule.training_coverage)
            self.cover_blocking_rules.append(blocking_rule)

    def fit(self, sdf: DataFrame) -> 'BlockLearner':
        """
        This method fits, i.e. learns, the blocking rules that are needed to cover `recall` percent of the training
        set pairs. The fitting is done by solving the set-cover problem. It is solved by using a greedy algorithm.

        Args:
            sdf: a labelled training set containing pairs.

        Returns:
            the object itself
        """
        # verify whether `row_id` and `label` are columns of `sdf_1`
        if 'row_id' not in sdf.columns:
            raise AssertionError('`row_id` is not present as a column of sdf_1')

        if 'label' not in sdf.columns:
            raise AssertionError('`label` is not present as a column of sdf_1')


        # determine the full set of pairs in the training data that have positive labels from active learning:
        sdf = (
            sdf
            .filter(F.col('label') == 1)
            .persist()  # break the lineage to avoid recomputing since sdf_1 is used many times during the fitting
        )
        self.full_set = set(sdf.select('row_id').toPandas()['row_id'])
        # determine the cardinality of the full_set, i.e. |full_set|:
        self.full_set_size = len(self.full_set)

        # calculate the training coverage for each blocking rule:
        self.blocking_rules = (
            list(
                map(
                    lambda x: x.calculate_training_set_coverage(sdf),
                    self.blocking_rules
                )
            )
        )

        # use a greedy set cover algorithm to select a subset of the blocking rules that cover `recall` * |full_set|
        self._greedy_set_coverage()
        if self.verbose:
            print('Blocking rules:', ", ".join([x.__repr__() for x in self.cover_blocking_rules]))
        self.fitted = True
        return self

    def _create_blocks(self, sdf: DataFrame) -> List[DataFrame]:
        """
        This method creates a list of blocked data. Blocked data is created by applying the learned blocking rules on
        the input dataframe.

        Args:
            sdf: dataframe containing records

        Returns:
            A list of blocked dataframes, i.e. list of dataframes containing block-keys
        """
        sdf_blocks = []
        for blocking_rule in self.cover_blocking_rules:
            sdf_blocks.append(blocking_rule.create_block_key(sdf))
        return sdf_blocks

    @staticmethod
    def _create_block_table(blocks: List[DataFrame]) -> DataFrame:
        """
        This method unifies the blocked data into a single dataframe.
        Args:
            blocks: containing blocked dataframes, i.e. dataframes containing block-keys

        Returns:
            a unified dataframe with all the block-keys
        """
        block_table = blocks[0]
        for block in blocks[1:]:
            block_table = block_table.unionByName(block)
        return block_table

    def transform(self, sdf_1: DataFrame, sdf_2: Optional[DataFrame] = None) -> Union[DataFrame,
                                                                                      Tuple[DataFrame, DataFrame]]:
        """
        This method adds the block-keys to the input dataframes. It applies all the learned blocking rules on the
        input data and unifies the results. The result of this method is/are the input dataframe(s) containing the
        block-keys from the learned blocking rules.

        Args:
            sdf_1: dataframe containing records
            sdf_2: dataframe containing records

        Returns:
            dataframe(s) containing block-keys from the learned blocking-rules
        """
        if not self.fitted:
            raise ValueError('BlockLearner is not yet fitted')

        sdf_1_blocks = self._create_blocks(sdf_1)
        sdf_1_blocks = self._create_block_table(sdf_1_blocks)

        if sdf_2:
            sdf_1_blocks = self.table_checkpointer(sdf_1_blocks, checkpoint_name='sdf_1_blocks')
            sdf_2_blocks = self._create_blocks(sdf_2)
            sdf_2_blocks = self._create_block_table(sdf_2_blocks)

            return sdf_1_blocks, sdf_2_blocks

        return sdf_1_blocks
