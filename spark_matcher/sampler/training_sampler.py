# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

import abc
from typing import List, Tuple, Optional, Union

import numpy as np
from pyspark.ml.feature import CountVectorizer, VectorAssembler, MinHashLSH
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql import DataFrame, functions as F, types as T

from spark_matcher.config import MINHASHING_MAXDF, MINHASHING_VOCABSIZE
from spark_matcher.table_checkpointer import TableCheckpointer


class Sampler:
    """
    Sampler base class to generate pairs for training.

    Args:
        col_names: list of column names to use for matching
        n_train_samples: nr of pair samples to be created for training
        table_checkpointer: table_checkpointer object to store cached tables
    """
    def __init__(self, col_names: List[str], n_train_samples: int, table_checkpointer: TableCheckpointer) -> DataFrame:
        self.col_names = col_names
        self.n_train_samples = n_train_samples
        self.table_checkpointer = table_checkpointer

    @abc.abstractmethod
    def create_pairs_table(self, sdf_1: DataFrame, sdf_2: DataFrame) -> DataFrame:
        pass


class HashSampler(Sampler):
    def __init__(self, table_checkpointer: TableCheckpointer, col_names: List[str], n_train_samples: int,
                 threshold: float = 0.5, num_hash_tables: int = 10) -> DataFrame:
        """
        Sampler class to generate pairs using MinHashLSH method by selecting pairs that are more likely to be a match.

        Args:
            threshold: for the distance of hashed pairs, values below threshold will be returned
                       (threshold being equal to 1.0 will return all pairs with at least one common shingle)
            num_hash_tables: number of hashes to be applied
            table_checkpointer: table_checkpointer object to store cached tables
        Returns:
            A spark dataframe which contains all selected pairs for training
        """
        super().__init__(col_names, n_train_samples, table_checkpointer)
        self.threshold = threshold
        self.num_hash_tables = num_hash_tables

    @staticmethod
    @F.udf(T.BooleanType())
    def is_non_zero_vector(vector):
        """
        Check if a vector has at least 1 non-zero entry. This function can deal with dense or sparse vectors. This is
        needed as the VectorAssembler can return dense or sparse vectors, dependent on what is more memory efficient.

        Args:
            vector: vector

        Returns:
            boolean whether a vector has at least 1 non-zero entry

        """
        if isinstance(vector, DenseVector):
            return bool(len(vector))
        if isinstance(vector, SparseVector):
            return vector.indices.size > 0

    def _vectorize(self, col_names: List[str], sdf_1: DataFrame, sdf_2: Optional[DataFrame] = None,
                   max_df: Union[float, int] = MINHASHING_MAXDF, vocab_size: int = MINHASHING_VOCABSIZE) -> Union[
        Tuple[DataFrame, DataFrame], DataFrame]:
        """
        This function is used to vectorize word based features. First, given dataframes are united to cover all feature
        space. Given columns are converted into a vector. `sdf_2` is only required for record matching, for
        deduplication only `sdf_1` is required. To prevent too many (redundant) matches on frequently occurring tokens,
        the maximum document frequency for vectorization is limited by maxDF.

        Args:
            col_names: list of column names to create feature vectors
            sdf_1: spark dataframe
            sdf_2: Optional: spark dataframe
            max_df: max document frequency to use for count vectorization
            vocab_size: vocabulary size of vectorizer

        Returns:
            Adds a 'features' column to given spark dataframes

        """
        if sdf_2:
            # creating a source column to separate tables after vectorization is completed
            sdf_1 = sdf_1.withColumn('source', F.lit('sdf_1'))
            sdf_2 = sdf_2.withColumn('source', F.lit('sdf_2'))
            sdf_merged = sdf_1.unionByName(sdf_2)
        else:
            sdf_merged = sdf_1

        for col in col_names:
            # converting strings into an array
            sdf_merged = sdf_merged.withColumn(f'{col}_array', F.split(F.col(col), ' '))

            # setting CountVectorizer
            # minDF is set to 2 because if a token occurs only once there is no other occurrence to match it to
            # the maximum document frequency may not be smaller than the minimum document frequency which is assured
            # below
            n_rows = sdf_merged.count()
            if isinstance(max_df, float) and (n_rows * max_df < 2):
                max_df = 10
            cv = CountVectorizer(binary=True, minDF=2, maxDF=max_df, vocabSize=vocab_size, inputCol=f"{col}_array",
                                 outputCol=f"{col}_vector")

            # creating vectors
            model = cv.fit(sdf_merged)
            sdf_merged = model.transform(sdf_merged)
        sdf_merged = self.table_checkpointer(sdf_merged, "cached_vectorized_table")

        # in case of col_names contains multiple columns, merge all vectors into one
        vec_assembler = VectorAssembler(outputCol="features")
        vec_assembler.setInputCols([f'{col}_vector' for col in col_names])

        sdf_merged = vec_assembler.transform(sdf_merged)

        # breaking the lineage is found to be required below
        sdf_merged = self.table_checkpointer(
            sdf_merged.filter(HashSampler.is_non_zero_vector('features')), 'minhash_vectorized')

        if sdf_2:
            sdf_1_vectorized = sdf_merged.filter(F.col('source') == 'sdf_1').select(*col_names, 'features')
            sdf_2_vectorized = sdf_merged.filter(F.col('source') == 'sdf_2').select(*col_names, 'features')
            return sdf_1_vectorized, sdf_2_vectorized
        sdf_1_vectorized = sdf_merged.select(*col_names, 'features')
        return sdf_1_vectorized

    @staticmethod
    def _apply_min_hash(sdf_1: DataFrame, sdf_2: DataFrame, col_names: List[str],
                        threshold: float, num_hash_tables: int):
        """
        This function is used to apply MinHasLSH technique to calculate the Jaccard distance between feature vectors.

        Args:
            sdf_1: spark dataframe with the `features` column and the column `row_number_1`
            sdf_2: spark dataframe with the `features` column and the column `row_number_2`
            col_names: list of column names to create feature vectors

        Returns:
            Creates one spark dataframe that contains all pairs that has a smaller Jaccard distance than threshold
        """
        mh = MinHashLSH(inputCol="features", outputCol="hashes", seed=42, numHashTables=num_hash_tables)

        model = mh.fit(sdf_1)
        model.transform(sdf_1)

        sdf_distances = (
            model
                .approxSimilarityJoin(sdf_1, sdf_2, threshold, distCol="JaccardDistance")
                .select(*[F.col(f'datasetA.{col}').alias(f'{col}_1') for col in col_names],
                        *[F.col(f'datasetB.{col}').alias(f'{col}_2') for col in col_names],
                        F.col('JaccardDistance').alias('distance'),
                        F.col('datasetA.row_number_1').alias('row_number_1'),
                        F.col('datasetB.row_number_2').alias('row_number_2'))
        )
        return sdf_distances

    def create_pairs_table(self, sdf_1: DataFrame, sdf_2: Optional[DataFrame] = None) -> DataFrame:
        """
        Create hashed pairs that are used for training. `sdf_2` is only required for record matching, for deduplication
        only `sdf_1` is required.

        Args:
            sdf_1: Spark dataframe containing the first table to with the input should be matched
            sdf_2: Optional: Spark dataframe containing the second table that should be matched to the first table

        Returns:
            Spark dataframe that contains sampled pairs selected with MinHashLSH technique

        """
        if sdf_2:
            sdf_1_vectorized, sdf_2_vectorized = self._vectorize(self.col_names, sdf_1, sdf_2)
            sdf_1_vectorized = self.table_checkpointer(
                                sdf_1_vectorized.withColumn('row_number_1', F.monotonically_increasing_id()),
                                checkpoint_name='sdf_1_vectorized')
            sdf_2_vectorized = self.table_checkpointer(
                                sdf_2_vectorized.withColumn('row_number_2', F.monotonically_increasing_id()),
                                checkpoint_name='sdf_2_vectorized')
        else:
            sdf_1_vectorized = self._vectorize(self.col_names, sdf_1)
            sdf_1_vectorized = self.table_checkpointer(
                                sdf_1_vectorized.withColumn('row_number_1', F.monotonically_increasing_id()),
                                checkpoint_name='sdf_1_vectorized')
            sdf_2_vectorized = sdf_1_vectorized.alias('sdf_2_vectorized') # matched with itself for deduplication
            sdf_2_vectorized = sdf_2_vectorized.withColumnRenamed('row_number_1', 'row_number_2')

        sdf_distances = self._apply_min_hash(sdf_1_vectorized, sdf_2_vectorized, self.col_names,
                                             self.threshold, self.num_hash_tables)

        # for deduplication we remove identical pairs like (a,a) and duplicates of pairs like (a,b) and (b,a)
        if not sdf_2:
            sdf_distances = sdf_distances.filter(F.col('row_number_1') < F.col('row_number_2'))

        hashed_pairs_table = (
            sdf_distances
            .sort('distance')
            .limit(self.n_train_samples)
            .drop('distance', 'row_number_1', 'row_number_2')
            )

        return hashed_pairs_table


class RandomSampler(Sampler):
    def __init__(self, table_checkpointer: TableCheckpointer, col_names: List[str], n_train_samples: int) -> DataFrame:
        """
        Sampler class to generate randomly selected pairs

        Returns:
            A spark dataframe which contains randomly selected pairs for training
        """
        super().__init__(col_names, n_train_samples, table_checkpointer)

    def create_pairs_table(self, sdf_1: DataFrame, sdf_2: Optional[DataFrame] = None) -> DataFrame:
        """
        Create random pairs that are used for training. `sdf_2` is only required for record matching, for deduplication
        only `sdf_1` is required.

        Args:
            sdf_1: Spark dataframe containing the first table to with the input should be matched
            sdf_2: Optional: Spark dataframe containing the second table that should be matched to the first table

        Returns:
            Spark dataframe that contains randomly sampled pairs

        """
        if sdf_2:
            return self._create_pairs_table_matcher(sdf_1, sdf_2)
        return self._create_pairs_table_deduplicator(sdf_1)

    def _create_pairs_table_matcher(self, sdf_1: DataFrame, sdf_2: DataFrame) -> DataFrame:
        """
        Create random pairs that are used for training.

        If the first table and the second table are equally large, we take the square root of `n_train_samples` from
        both tables to create a pairs table containing `n_train_samples` rows. If one of the table is much smaller than
        the other, all rows of the smallest table are taken and the number of the sample from the larger table is chosen
        such that the total number of pairs is `n_train_samples`.

        Args:
            sdf_1: Spark dataframe containing the first table to with the input should be matched
            sdf_2: Spark dataframe containing the second table that should be matched to the first table

        Returns:
            Spark dataframe with randomly sampled pairs to compared during training
        """
        sdf_1_count, sdf_2_count = sdf_1.count(), sdf_2.count()

        sample_size_small_table = min([int(self.n_train_samples ** 0.5), min(sdf_1_count, sdf_2_count)])
        sample_size_large_table = self.n_train_samples // sample_size_small_table

        if sdf_1_count > sdf_2_count:
            fraction_sdf_1 = sample_size_large_table / sdf_1_count
            fraction_sdf_2 = sample_size_small_table / sdf_2_count
            # below the `fraction` is slightly increased (but capped at 1.) and a `limit()` is applied to assure that
            # the exact number of required samples is obtained
            sdf_1_sample = (sdf_1.sample(withReplacement=False, fraction=min([1., 1.5 * fraction_sdf_1]))
                            .limit(sample_size_large_table))
            sdf_2_sample = (sdf_2.sample(withReplacement=False, fraction=min([1., 1.5 * fraction_sdf_2]))
                            .limit(sample_size_small_table))
        else:
            fraction_sdf_1 = sample_size_small_table / sdf_1_count
            fraction_sdf_2 = sample_size_large_table / sdf_2_count
            sdf_1_sample = (sdf_1.sample(withReplacement=False, fraction=min([1., 1.5 * fraction_sdf_1]))
                            .limit(sample_size_small_table))
            sdf_2_sample = (sdf_2.sample(withReplacement=False, fraction=min([1., 1.5 * fraction_sdf_2]))
                            .limit(sample_size_large_table))

        for col in self.col_names:
            sdf_1_sample = sdf_1_sample.withColumnRenamed(col, col + "_1")
            sdf_2_sample = sdf_2_sample.withColumnRenamed(col, col + "_2")

        random_pairs_table = sdf_1_sample.crossJoin(sdf_2_sample)

        return random_pairs_table

    def _create_pairs_table_deduplicator(self, sdf: DataFrame) -> DataFrame:
        """
        Create randomly selected pairs for deduplication.

        Args:
            sdf: Spark dataframe containing rows from which pairs need to be created

        Returns:
            Spark dataframe with the randomly selected pairs

        """
        n_samples_required = int(1.5 * np.ceil((self.n_train_samples * 2) ** 0.5))
        fraction_samples_required = min([n_samples_required / sdf.count(), 1.])
        sample = sdf.sample(withReplacement=False, fraction=fraction_samples_required)
        sample = self.table_checkpointer(sample.withColumn('row_id', F.monotonically_increasing_id()),
                                                          checkpoint_name='random_pairs_deduplicator')
        sample_1, sample_2 = sample, sample
        for col in self.col_names + ['row_id']:
            sample_1 = sample_1.withColumnRenamed(col, col + "_1")
            sample_2 = sample_2.withColumnRenamed(col, col + "_2")

        pairs_table = (sample_1
                       .crossJoin(sample_2)
                       .filter(F.col('row_id_1') < F.col('row_id_2'))
                       .limit(self.n_train_samples)
                       .drop('row_id_1', 'row_id_2'))

        return pairs_table
