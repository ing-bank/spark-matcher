# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from typing import List, Tuple

from graphframes import GraphFrame
from pyspark.sql import functions as F, types as T, DataFrame, Window

from spark_matcher.table_checkpointer import TableCheckpointer


class ConnectedComponentsCalculator:

    def __init__(self, scored_pairs_table: DataFrame, max_edges_clustering: int,
                 edge_filter_thresholds: List[float], table_checkpointer: TableCheckpointer):
        self.scored_pairs_table = scored_pairs_table
        self.max_edges_clustering = max_edges_clustering
        self.edge_filter_thresholds = edge_filter_thresholds
        self.table_checkpointer = table_checkpointer

    @staticmethod
    def _create_graph(scores_table: DataFrame) -> GraphFrame:
        """
        This function creates a graph where each row-number is a vertex and each similarity score between a pair of
        row-numbers represents an edge. This Graph is used as input for connected components calculations to determine
        the subgraphs of linked pairs.

        Args:
            scores_table: a pairs table with similarity scores for each pair
        Returns:
            a GraphFrames graph object representing the row-numbers and the similarity scores between them as a graph
        """

        vertices_1 = scores_table.select('row_number_1').withColumnRenamed('row_number_1', 'id')
        vertices_2 = scores_table.select('row_number_2').withColumnRenamed('row_number_2', 'id')
        vertices = vertices_1.unionByName(vertices_2).drop_duplicates()

        edges = (
            scores_table
                .select('row_number_1', 'row_number_2', 'score')
                .withColumnRenamed('row_number_1', 'src')
                .withColumnRenamed('row_number_2', 'dst')
        )
        return GraphFrame(vertices, edges)

    def _calculate_connected_components(self, scores_table: DataFrame, checkpoint_name: str) -> DataFrame:
        """
        This function calculates the connected components (i.e. the subgraphs) of a graph of scored pairs.
        The result of the connected-components algorithm is cached and saved.

        Args:
            scores_table: a pairs table with similarity scores for each pair
        Returns:
            a spark dataframe containing the connected components of a graph of scored pairs.
        """
        graph = self._create_graph(scores_table)
        connected_components = graph.connectedComponents()
        return self.table_checkpointer(connected_components, checkpoint_name)

    @staticmethod
    def _add_component_id_to_scores_table(scores_table: DataFrame, connected_components: DataFrame) -> DataFrame:
        """
        This function joins the initial connected-component identifiers to the scored pairs table. For each scored pair,
        a component identifier indicating to which subgraph a pair belongs is added.

        Args:
            scores_table: a pairs table with similarity scores for each pair
            connected_components: a spark dataframe containing the result of the connected components algorithm
        Returns:
            the scores_table with an identifier that indicates to which component a pair belongs
        """
        scores_table_with_component_ids = (
            scores_table
            .join(connected_components, on=scores_table['row_number_1']==connected_components['id'], how='left')
            .withColumnRenamed('component', 'component_1')
            .drop('id')
            .join(connected_components, on=scores_table['row_number_2']==connected_components['id'], how='left')
            .withColumnRenamed('component', 'component_2')
            .drop('id')
        )

        scores_table_with_component_ids = (
            scores_table_with_component_ids
            .withColumnRenamed('component_1', 'component')
            .drop('component_2')
        )
        return scores_table_with_component_ids

    @staticmethod
    def _add_big_components_info(scored_pairs_w_components: DataFrame, component_cols: List[str], max_size: int) -> \
    Tuple[DataFrame, bool]:
        """
        This function adds information if there are components in the scored_pairs table that contain more pairs than
        the `max_size'. If there is a component that is too big, the function will identify this and will indicate that
        filter iterations are required via the `continue_iteration` value.

        Args:
            scored_pairs_w_components: a pairs table with similarity scores for each pair and component identifiers
            component_cols: a list of columns that represent the composite key that indicate connected-components
            after one or multiple iterations
            max_size: a number indicating the maximum size of the number of pairs that are in a connected component
            in the scores table
        Returns:
            the scores table with a column indicating whether a pair belongs to a big component
            and a boolean indicating whether there are too big components and thus whether more iterations are required.
        """
        window = Window.partitionBy(*component_cols)
        big_components_info = (
            scored_pairs_w_components
                .withColumn('constant', F.lit(1))
                .withColumn('component_size', F.count('constant').over(window))
                .drop('constant')
                .withColumn('is_in_big_component', F.col('component_size') > max_size)
                .drop('component_size')
        )
        continue_iteration = False
        if big_components_info.filter(F.col('is_in_big_component')).count() > 0:
            continue_iteration = True
        return big_components_info, continue_iteration

    @staticmethod
    def _add_low_score_info(scored_pairs: DataFrame, threshold) -> DataFrame:
        """
        This function adds a column that indicates whether a similarity score between pairs is lower than a `threshold`
        value.

        Args:
            scored_pairs: a pairs table with similarity scores for each pair
            threshold: a value indicating the treshold value for a similarity score
        Returns:
            the scores table with an addiction column indicating whether a pair has a score lower than the `threshold`
        """
        scored_pairs = (
            scored_pairs
                .withColumn('is_lower_than_threshold', F.col('score') <threshold)
        )
        return scored_pairs

    @staticmethod
    def _join_connected_components_iteration_results(scored_pairs: DataFrame, connected_components: DataFrame,
                                                     iteration_number: int) -> DataFrame:
        """
        This function joins the the connected component results for each iteration to the scores table and prunes the
        scores table by removing edges that belong to a too big component and have a similarity score that is lower than
        the current threshold.

        Args:
            scored_pairs: a pairs table with similarity scores for each pair for an iteration
            connected_components: a spark dataframe containing the result of the connected components algorithm for
            an iteration
            iteration_number: a counter indicating which iteration it is
        Returns:
            a scores table with the component identifiers for an iteration and without edges that belonged to too big
            components
            that had scores that were below the current threshold.
        """
        # join the connected components results to the scored-pairs on `row_number_1`
        result = (
            scored_pairs
                .join(connected_components, on=scored_pairs['row_number_1'] == connected_components['id'], how='left')
                .drop('id')
                .withColumnRenamed('component', 'component_1')
                .persist())

        # join the connected components results to the scored-pairs on `row_number_2`
        result = (
            result
                .join(connected_components, on=scored_pairs['row_number_2'] == connected_components['id'], how='left')
                .drop('id')
                .withColumnRenamed('component', 'component_2')
                .filter((F.col('component_1') == F.col('component_2')) | (F.col('component_1').isNull() & F.col(
                'component_2').isNull()))
                .withColumnRenamed('component_1', f'component_iteration_{iteration_number}')
                .drop('component_2')
                # add a default value for rows that were not part of the iteration, i.e. -1 since component ids' are
                # always positive integers
                .fillna(-1, subset=[f'component_iteration_{iteration_number}'])
                .filter((F.col('is_in_big_component') & ~F.col('is_lower_than_threshold')) |
                        (~F.col('is_in_big_component')))
                .drop('is_in_big_component', 'is_lower_than_threshold')
        )
        return result

    @staticmethod
    def _add_component_id(sdf: DataFrame, iteration_count: int) -> DataFrame:
        """
        This function adds a final `component_id` after all iterations that can be used as a groupby-key to
        distribute the deduplication. The `component_id` is a hashed value of the concatenation with '_' seperation
        of all component identifiers of the iterations. The sha2 hash function is used order to make the likelihood of
        hash collisions negligibly small.

        Args:
            sdf: a dataframe containing the scored pairs
            iteration_count: a number indicating the final iteration number that is used.
        Returns:
            the final scores-table with a component-id that can be used as groupby-key to distribute the deduplication.
        """
        if iteration_count > 0:
            component_id_cols = ['initial_component'] + [f'component_iteration_{i}' for i in
                                                         range(1, iteration_count + 1)]
            sdf = sdf.withColumn('component_id', F.sha2(F.concat_ws('_', *component_id_cols), numBits=256))
        else:
            # use the initial component as component_id and cast the value to string to match with the return type of
            # the sha2 algorithm
            sdf = (
                sdf
                .withColumn('component_id', F.col('initial_component').cast(T.StringType()))
                .drop('initial_component')
                )
        return sdf

    def _create_component_ids(self) -> DataFrame:
        """
        This function wraps the other methods in this class and performs the connected-component calculations and the
        edge filtering if required.

        Returns:
            a scored-pairs table with a component identifier that can be used as groupby-key to distribute the
            deduplication.
        """
        # calculate the initial connected components
        connected_components = self._calculate_connected_components(self.scored_pairs_table,
                                                                    "cached_connected_components_table_initial")
        # add the component ids to the scored-pairs table
        self.scored_pairs_table = self._add_component_id_to_scores_table(self.scored_pairs_table, connected_components)
        self.scored_pairs_table = self.scored_pairs_table.withColumnRenamed('component', 'initial_component')

        #checkpoint the results
        self.scored_pairs_table = self.table_checkpointer(self.scored_pairs_table,
                                                          "cached_scores_with_components_table_initial")

        component_columns = ['initial_component']
        iteration_count = 0
        for i in range(1, len(self.edge_filter_thresholds) + 1):
            # set and update the threshold
            threshold = self.edge_filter_thresholds[i - 1]

            self.scored_pairs_table, continue_iter = self._add_big_components_info(self.scored_pairs_table,
                                                                                   component_columns,
                                                                                   self.max_edges_clustering)
            if not continue_iter:
                # no further iterations are required
                break

            self.scored_pairs_table = self._add_low_score_info(self.scored_pairs_table, threshold)
            # recalculate the connected_components
            iteration_input_sdf = (
                self.scored_pairs_table
                .filter((F.col('is_in_big_component') & ~F.col('is_lower_than_threshold')))
            )

            print(f"calculate connected components iteration {i} with edge filter threshold {threshold}")
            connected_components_iteration = self._calculate_connected_components(iteration_input_sdf,
                                                                f'cached_connected_components_table_iteration_{i}')

            self.scored_pairs_table = (
                self
                    ._join_connected_components_iteration_results(self.scored_pairs_table,
                                                                  connected_components_iteration, i)
            )

            # checkpoint the result to break the lineage after each iteration and to inspect intermediate results
            self.scored_pairs_table = self.table_checkpointer(self.scored_pairs_table,
                                                                f'cached_scores_with_components_table_iteration_{i}')

            component_columns.append(f'component_iteration_{i}')
            iteration_count += 1

        # final check to see if there were sufficient iterations to reduce the size of the big components if all
        # specified iterations are used to reduce the component size
        if iteration_count == len(self.edge_filter_thresholds):
            self.scored_pairs_table, continue_iter = self._add_big_components_info(self.scored_pairs_table,
                                                                                   component_columns,
                                                                                   self.max_edges_clustering)
            if continue_iter:
                print("THE EDGE-FILTER-THRESHOLDS ARE NOT ENOUGH TO SUFFICIENTLY REDUCE THE SIZE OF THE BIG COMPONENTS")

        # add the final component-id that can be used as groupby key for distributed deduplication
        self.scored_pairs_table = self._add_component_id(self.scored_pairs_table, iteration_count)

        return self.table_checkpointer(self.scored_pairs_table, 'cached_scores_with_components_table')
