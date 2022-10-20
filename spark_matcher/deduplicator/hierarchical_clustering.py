# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

from collections import defaultdict
from typing import List, Dict, Callable

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy


def _perform_clustering(component: pd.DataFrame, threshold: float, linkage_method: str):
    """
    Apply hierarchical clustering to scored_pairs_table with component_ids
    Args:
        component: pandas dataframe containing all pairs and the similarity scores for a connected component
        threshold: threshold to apply in hierarchical clustering
        linkage_method: linkage method to apply in hierarchical clustering
    Returns:
        Generator that contains tuples of ids and scores
    """

    distance_threshold = 1 - threshold
    if len(component) > 1:
        i_to_id, condensed_distances = _get_condensed_distances(component)

        linkage = hierarchy.linkage(condensed_distances, method=linkage_method)
        partition = hierarchy.fcluster(linkage, t=distance_threshold, criterion='distance')

        clusters: Dict[int, List[int]] = defaultdict(list)

        for i, cluster_id in enumerate(partition):
            clusters[cluster_id].append(i)

        for cluster in clusters.values():
            if len(cluster) > 1:
                yield tuple(i_to_id[i] for i in cluster), None
    else:
        ids = np.array([int(component['row_number_1']), int(component['row_number_2'])])
        score = float(component['score'])
        if score > threshold:
            yield tuple(ids), (score,) * 2


def _convert_data_to_adjacency_matrix(component: pd.DataFrame):
    """
    This function converts a pd.DataFrame to a numpy adjacency matrix
    Args:
        component: pd.DataFrame
    Returns:
        index of elements of the components and a numpy adjacency matrix
    """
    def _get_adjacency_matrix(df, col1, col2, score_col:str = 'score'):
        df = pd.crosstab(df[col1], df[col2], values=df[score_col], aggfunc='max')
        idx = df.columns.union(df.index)
        df = df.reindex(index = idx, columns=idx, fill_value=0).fillna(0)
        return df

    a_to_b = _get_adjacency_matrix(component, "row_number_1", "row_number_2")
    b_to_a = _get_adjacency_matrix(component, "row_number_2", "row_number_1")

    symmetric_adjacency_matrix = a_to_b + b_to_a

    return symmetric_adjacency_matrix.index, np.array(symmetric_adjacency_matrix)


def _get_condensed_distances(component: pd.DataFrame):
    """
    Converts the pairwise list of distances to "condensed distance matrix" required by the hierarchical clustering
    algorithms. Also return a dictionary that maps the distance matrix to the ids.

    Args:
        component: pandas dataframe containing all pairs and the similarity scores for a connected component

    Returns:
        condensed distances and a dict with ids
    """

    i_to_id, adj_matrix = _convert_data_to_adjacency_matrix(component)
    distances = (np.ones_like(adj_matrix) - np.eye(len(adj_matrix))) - adj_matrix
    return dict(enumerate(i_to_id)), ssd.squareform(distances)


def _convert_dedupe_result_to_pandas_dataframe(dedupe_result: List, component_id: int) -> pd.DataFrame:
    """
    Function to convert the dedupe result into a pandas dataframe.
    E.g.
        dedupe_result = [((1, 2), array([0.96, 0.96])), ((3, 4, 5), array([0.95, 0.95, 0.95]))]

        returns

        | row_number | entity_identifier |
        | ---------- | ------------------- |
        | 1          | 1                   |
        | 2          | 1                   |
        | 3          | 2                   |
        | 4          | 2                   |
        | 5          | 2                   |

    Args:
        dedupe_result: the result with the deduplication results from the clustering
    Returns:
        pandas dataframe with row_number and entity_identifier
    """
    if len(dedupe_result) == 0:
        return pd.DataFrame(data={'row_number': [], 'entity_identifier': []})

    entity_identifier = 0
    df_list = []

    for ids, _ in dedupe_result:
        df_ = pd.DataFrame(data={'row_number': list(ids), 'entity_identifier': f"{component_id}_{entity_identifier}"})
        df_list.append(df_)
        entity_identifier += 1
    return pd.concat(df_list)


def apply_deduplication(cluster_score_threshold: float, cluster_linkage_method: str) -> Callable:
    """
    This function is a wrapper function to parameterize the _apply_deduplucation function with extra parameters for
    the cluster_score_threshold and the linkage method.
    Args:
        cluster_score_threshold: a float in [0,1]
        cluster_linkage_method: a string indicating the linkage method to be used for hierarchical clustering
    Returns:
        a function, i.e. _apply_deduplication, that can be called as a Pandas udf
    """
    def _apply_deduplication(component: pd.DataFrame) -> pd.DataFrame:
        """
        This function applies deduplication on a component, i.e. a subgraph calculated by the connected components
        algorithm. This function is applied to a spark dataframe in a pandas udf to distribute the deduplication
        over a Spark cluster, component by component.
        Args:
            component: a pandas Dataframe
        Returns:
            a pandas Dataframe with the results from the hierarchical clustering of the deduplication
        """
        component_id = component['component_id'][0]

        # perform the clustering:
        component = list(_perform_clustering(component, cluster_score_threshold, cluster_linkage_method))

        # convert the results to a dataframe:
        return _convert_dedupe_result_to_pandas_dataframe(component, component_id)

    return _apply_deduplication
