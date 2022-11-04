# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import Optional, List, Dict

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T
from sklearn.exceptions import NotFittedError
from scipy.cluster import hierarchy

from spark_matcher.blocker.blocking_rules import BlockingRule
from spark_matcher.deduplicator.connected_components_calculator import ConnectedComponentsCalculator
from spark_matcher.deduplicator.hierarchical_clustering import apply_deduplication
from spark_matcher.matching_base.matching_base import MatchingBase
from spark_matcher.scorer.scorer import Scorer
from spark_matcher.table_checkpointer import TableCheckpointer


class Deduplicator(MatchingBase):
    """
    Deduplicator class to apply deduplication. Provide either the column names `col_names` using the default string
    similarity metrics or explicitly define the string similarity metrics in a dict `field_info` as in the example
    below. If `blocking_rules` is left empty, default blocking rules are used. Otherwise, provide blocking rules as
    a list containing `BlockingRule` instances (see example below). The number of perfect matches used during
    training is set by `n_perfect_train_matches`.

    E.g.:

    from spark_matcher.blocker.blocking_rules import FirstNChars

    myDeduplicator = Deduplicator(spark_session, field_info={'name':[metric_function_1, metric_function_2],
                                                            'address:[metric_function_1, metric_function_3]},
                                                            blocking_rules=[FirstNChars('name', 3)])

    Args:
        spark_session: Spark session
        col_names: list of column names to use for matching
        field_info: dict of column names as keys and lists of string similarity metrics as values
        blocking_rules: list of `BlockingRule` instances
        table_checkpointer: pointer object to store cached tables
        checkpoint_dir: checkpoint directory if provided
        n_train_samples: nr of pair samples to be created for training
        ratio_hashed_samples: ratio of hashed samples to be created for training, rest is sampled randomly
        n_perfect_train_matches: nr of perfect matches used for training
        scorer: a Scorer object used for scoring pairs
        verbose: sets verbosity
        max_edges_clustering: max number of edges per component that enters clustering
        edge_filter_thresholds: list of score thresholds to use for filtering when components are too large
        cluster_score_threshold: threshold value between [0.0, 1.0], only pairs are put together in clusters if
                                 cluster similarity scores are >= cluster_score_threshold
        cluster_linkage_method: linkage method to be used within hierarchical clustering, can take values such as
        'centroid', 'single', 'complete', 'average', 'weighted', 'median', 'ward' etc.
    """
    def __init__(self, spark_session: SparkSession, col_names: Optional[List[str]] = None,
                 field_info: Optional[Dict] = None, blocking_rules: Optional[List[BlockingRule]] = None,
                 blocking_recall: float = 1.0, table_checkpointer: Optional[TableCheckpointer] = None,
                 checkpoint_dir: Optional[str] = None, n_perfect_train_matches=1, n_train_samples: int = 100_000,
                 ratio_hashed_samples: float = 0.5, scorer: Optional[Scorer] = None, verbose: int = 0,
                 max_edges_clustering: int = 500_000,
                 edge_filter_thresholds: List[float] = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
                 cluster_score_threshold: float = 0.5, cluster_linkage_method: str = "centroid"):

        super().__init__(spark_session, table_checkpointer, checkpoint_dir, col_names, field_info, blocking_rules,
                         blocking_recall, n_perfect_train_matches, n_train_samples, ratio_hashed_samples, scorer,
                         verbose)

        self.fitted_ = False
        self.max_edges_clustering = max_edges_clustering
        self.edge_filter_thresholds = edge_filter_thresholds
        self.cluster_score_threshold = cluster_score_threshold
        if cluster_linkage_method not in list(hierarchy._LINKAGE_METHODS.keys()):
            raise ValueError(f"Invalid cluster_linkage_method: {cluster_linkage_method}")
        self.cluster_linkage_method = cluster_linkage_method
        # set the checkpoints directory for graphframes
        self.spark_session.sparkContext.setCheckpointDir('checkpoints/')

    def _create_predict_pairs_table(self, sdf_blocked: DataFrame) -> DataFrame:
        """
        This method performs an alias self-join on `sdf` to create the pairs table for prediction. `sdf` is joined to
        itself based on the `block_key` column. The result of this join is a pairs table.
        """
        sdf_blocked_1 = self._add_suffix_to_col_names(sdf_blocked, 1)
        sdf_blocked_1 = sdf_blocked_1.withColumnRenamed('row_number', 'row_number_1')
        sdf_blocked_2 = self._add_suffix_to_col_names(sdf_blocked, 2)
        sdf_blocked_2 = sdf_blocked_2.withColumnRenamed('row_number', 'row_number_2')

        pairs = (
            sdf_blocked_1
                .join(sdf_blocked_2, on='block_key', how='inner')
                .filter(F.col("row_number_1") < F.col("row_number_2"))
                .drop_duplicates(subset=[col + "_1" for col in self.col_names] + [col + "_2" for col in self.col_names])
        )
        return pairs

    def _map_distributed_identifiers_to_long(self, clustered_results: DataFrame) -> DataFrame:
        """
        Method to add a unique `entity_identifier` to the results from clustering

        Args:
            clustered_results: results from clustering
        Returns:
            Spark dataframe containing `row_number` and `entity_identifier`
        """
        long_entity_ids = (
            clustered_results
                .select('entity_identifier')
                .drop_duplicates()
                .withColumn('long_entity_identifier', F.monotonically_increasing_id())
        )
        long_entity_ids = self.table_checkpointer(long_entity_ids, 'cached_long_ids_table')

        clustered_results = (
            clustered_results
            .join(long_entity_ids, on='entity_identifier', how='left')
            .drop('entity_identifier')
            .withColumnRenamed('long_entity_identifier', 'entity_identifier')
        )
        return clustered_results

    def _distributed_deduplication(self, scored_pairs_with_components: DataFrame) -> DataFrame:
        schema = T.StructType([T.StructField('row_number', T.LongType(), True),
                               T.StructField('entity_identifier', T.StringType(), True)])

        clustered_results = (
            scored_pairs_with_components
            .select('row_number_1', 'row_number_2', 'score', 'component_id')
            .groupby('component_id')
            .applyInPandas(apply_deduplication(self.cluster_score_threshold, self.cluster_linkage_method) , schema=schema)
        )
        return self.table_checkpointer(clustered_results, "cached_clustered_results_table")

    @staticmethod
    def _add_singletons_entity_identifiers(result_sdf: DataFrame) -> DataFrame:
        """
        Function to add entity_identifier to entities (singletons) that are not combined with other entities into a
        deduplicated entity. If there are no singletons, the input table will be returned as it is.

        Args:
            result_sdf: Spark dataframe containing the result of deduplication where entities that are not deduplicated
            have a missing value in the `entity_identifier` column.

        Returns:
            Spark dataframe with `entity_identifier` values for all entities

        """
        if result_sdf.filter(F.col('entity_identifier').isNull()).count() == 0:
            return result_sdf
        start_cluster_id = (
            result_sdf
                .filter(F.col('entity_identifier').isNotNull())
                .select(F.max('entity_identifier'))
                .first()[0])

        singletons_entity_identifiers = (
            result_sdf
                .filter(F.col('entity_identifier').isNull())
                .select('row_number')
                .rdd
                .zipWithIndex()
                .toDF()
                .withColumn('row_number', F.col('_1').getItem("row_number"))
                .drop("_1")
                .withColumn('entity_identifier_singletons', F.col('_2') + start_cluster_id + 1)
                .drop("_2"))

        result_sdf = (
            result_sdf
                .join(singletons_entity_identifiers, on='row_number', how='left')
                .withColumn('entity_identifier',
                            F.when(F.col('entity_identifier').isNull(),
                                F.col('entity_identifier_singletons'))
                            .otherwise(F.col('entity_identifier')))
                .drop('entity_identifier_singletons'))

        return result_sdf

    def _create_deduplication_results(self, clustered_results: DataFrame, entities: DataFrame) -> DataFrame:
        """
        Joins deduplication results back to entity_table and adds identifiers to rows that are not deduplicated with
        other rows
        """
        deduplication_results = (
            entities
            .join(clustered_results, on='row_number', how='left')
        )

        deduplication_results = (
                self._add_singletons_entity_identifiers(deduplication_results)
                .drop('row_number', 'block_key')
        )
        return deduplication_results

    def _get_large_clusters(self, pairs_with_components: DataFrame) -> DataFrame:
        """
        Components that are too large after iteratively removing edges with a similarity score lower than the
        thresholds in `edge_filter_thresholds`, are considered to be one entity. For these the `component_id` is
        temporarily used as an `entity_identifier`.
        """
        large_component = (pairs_with_components.filter(F.col('is_in_big_component'))
                           .withColumn('entity_identifier', F.col('component_id')))

        large_cluster = (large_component.select('row_number_1', 'entity_identifier')
                         .withColumnRenamed('row_number_1', 'row_number')
                         .unionByName(large_component.select('row_number_2', 'entity_identifier')
                                      .withColumnRenamed('row_number_2', 'row_number'))
                         .drop_duplicates()
                         .persist())
        return large_cluster

    def predict(self, sdf: DataFrame, threshold: float = 0.5):
        """
        Method to predict on data used for training or new data.

        Args:
            sdf: table to be applied entity deduplication
            threshold: probability threshold for similarity score

        Returns:
            Spark dataframe with the deduplication result

        """
        if not self.fitted_:
            raise NotFittedError('The Deduplicator instance is not fitted yet. Call `fit` and train the instance.')

        sdf = self._simplify_dataframe_for_matching(sdf)
        sdf = sdf.withColumn('row_number', F.monotonically_increasing_id())
        # make sure the `row_number` is a fixed value. See the docs of `monotonically_increasing_id`, the function
        # depends on the partitioning of the table. Not fixing/storing the result will cause problems when the table is
        # recalculated under the hood during other operations. Saving it to disk and reading it in, breaks lineage and
        # forces the result to be deterministic in subsequent operations.
        sdf = self.table_checkpointer(sdf, "cached_entities_numbered_table")

        sdf_blocked = self.table_checkpointer(self.blocker.transform(sdf), "cached_block_table")
        pairs_table = self.table_checkpointer(self._create_predict_pairs_table(sdf_blocked), "cached_pairs_table")
        metrics_table = self.table_checkpointer(self._calculate_metrics(pairs_table), "cached_metrics_table")

        scores_table = (
            metrics_table
            .withColumn('score', self.scoring_learner.predict_proba(metrics_table['similarity_metrics']))
        )
        scores_table = self.table_checkpointer(scores_table, "cached_scores_table")

        scores_table_filtered = (
            scores_table
            .filter(F.col('score') >= threshold)
            .drop('block_key', 'similarity_metrics')
            )

        # create the subgraphs by using the connected components algorithm
        connected_components_calculator = ConnectedComponentsCalculator(scores_table_filtered,
                                                                        self.max_edges_clustering,
                                                                        self.edge_filter_thresholds,
                                                                        self.table_checkpointer)

        # calculate the component id for each scored-pairs
        scored_pairs_with_component_ids = connected_components_calculator._create_component_ids()

        large_cluster = self._get_large_clusters(scored_pairs_with_component_ids)

        # apply the distributed deduplication to components that are sufficiently small
        clustered_results = self._distributed_deduplication(
            scored_pairs_with_component_ids.filter(~F.col('is_in_big_component')))

        all_results = large_cluster.unionByName(clustered_results)

        # assign a unique entity_identifier to all components
        all_results_long = self._map_distributed_identifiers_to_long(all_results)
        all_results_long = self.table_checkpointer(all_results_long, "cached_clustered_results_with_long_ids_table")

        deduplication_results = self._create_deduplication_results(all_results_long, sdf)

        return deduplication_results
