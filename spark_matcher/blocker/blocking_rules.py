# Authors: Ahmet Bayraktar
#          Stan Leisink
#          Frits Hermans

import abc

from pyspark.sql import Column, DataFrame, functions as F, types as T


class BlockingRule(abc.ABC):
    """
    Abstract class for blocking rules. This class contains all the base functionality for blocking rules.

    Attributes:
        blocking_column: the column on which the `BlockingRule` is applied
    """

    def __init__(self, blocking_column: str):
        self.blocking_column = blocking_column
        self.training_coverage = None
        self.training_coverage_size = None

    @abc.abstractmethod
    def _blocking_rule(self, c: Column) -> Column:
        """
        Abstract method for a blocking-rule

        Args:
            c: a Column

        Returns:
            A Column with a blocking-rule result
        """
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """
        Abstract method for a class representation

        Returns:
            return a string that represents the blocking-rule

        """
        pass

    def _apply_blocking_rule(self, c: Column) -> Column:
        """
        This method applies the blocking rule on an input column and adds a unique representation id to make sure the
        block-key is unique. Uniqueness is important to avoid collisions between block-keys, e.g. a blocking-rule that
        captures the first string character can return the same as a blocking-rule that captures the last character. To
        avoid this, the blocking-rule results is concatenated with the __repr__ method.

        Args:
            c: a Column

        Returns:
            a Column with a block_key

        """
        return F.concat(F.lit(f'{self.__repr__()}:'), self._blocking_rule(c))

    def create_block_key(self, sdf: DataFrame) -> DataFrame:
        """
        This method calculates and adds the block-key column to the input dataframe

        Args:
            sdf: a dataframe with records that need to be matched

        Returns:
            the dataframe with the block-key column
        """
        return sdf.withColumn('block_key', self._apply_blocking_rule(sdf[f'{self.blocking_column}']))

    def _create_training_block_keys(self, sdf: DataFrame) -> DataFrame:
        """
        This method is used to create block-keys on a training dataframe

        Args:
            sdf: a dataframe containing record pairs for training

        Returns:
            the dataframe containing block-keys for the record pairs
        """
        return (
            sdf
                .withColumn('block_key_1', self._apply_blocking_rule(sdf[f'{self.blocking_column}_1']))
                .withColumn('block_key_2', self._apply_blocking_rule(sdf[f'{self.blocking_column}_2']))
        )

    @staticmethod
    def _compare_and_filter_keys(sdf: DataFrame) -> DataFrame:
        """
        This method is used to compare block-keys of record pairs and subsequently filter record pairs in the training
        dataframe that have identical block-keys

        Args:
            sdf: a dataframe containing record pairs for training with their block-keys

        Returns:
            the filtered dataframe containing only record pairs with identical block-keys
        """
        return (
            sdf
                # split on `:` since this separates the `__repr__` from the blocking_rule result
                .withColumn('_blocking_result_1', F.split(F.col('block_key_1'), ":").getItem(1))
                .withColumn('_blocking_result_2', F.split(F.col('block_key_2'), ":").getItem(1))
                .filter(
                (F.col('block_key_1') == F.col('block_key_2')) &
                ((F.col('_blocking_result_1') != '') & (F.col('_blocking_result_2') != ''))
            )
                .drop('_blocking_result_1', '_blocking_result_2')
        )

    @staticmethod
    def _length_check(c: Column, n: int, word_count: bool = False) -> Column:
        """
        This method checks the length of the created block key.

        Args:
            c: block key to check
            n: given length of the string
            word_count: whether to check the string length or word count
        Returns:
            the block key if it is not shorter than the given length, otherwise returns None
        """
        if word_count:
            return F.when(F.size(c) >= n, c).otherwise(None)

        return F.when(F.length(c) == n, c).otherwise(None)

    def calculate_training_set_coverage(self, sdf: DataFrame) -> 'BlockingRule':
        """
        This method calculate the set coverage of the blocking rule on the training pairs. The set coverage of the rule
        is determined by looking at how many record pairs in the training set end up in the same block. This coverage
        is used in the BlockLearner to sort blocking rules in the greedy set_covering algorithm.
        Args:
            sdf: a dataframe containing record pairs for training

        Returns:
            The object itself
        """
        sdf = self._create_training_block_keys(sdf)

        sdf = self._compare_and_filter_keys(sdf)

        self.training_coverage = set(
            sdf
                .agg(F.collect_set('row_id').alias('training_coverage'))
                .collect()[0]['training_coverage']
        )

        self.training_coverage_size = len(self.training_coverage)
        return self


# define the concrete blocking rule examples:

class FirstNChars(BlockingRule):
    def __init__(self, blocking_column: str, n: int = 3):
        super().__init__(blocking_column)
        self.n = n

    def __repr__(self):
        return f"first_{self.n}_characters_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        key = F.substring(c, 0, self.n)
        return self._length_check(key, self.n)


class FirstNCharsLastWord(BlockingRule):
    def __init__(self, blocking_column: str, n: int = 3, remove_non_alphanumerical=False):
        super().__init__(blocking_column)
        self.n = n
        self.remove_non_alphanumerical = remove_non_alphanumerical

    def __repr__(self):
        return f"first_{self.n}_characters_last_word_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        if self.remove_non_alphanumerical:
            c = F.regexp_replace(c, r'\W+', ' ')
        tokens = F.split(c, r'\s+')
        last_word = F.element_at(tokens, -1)
        key = F.substring(last_word, 1, self.n)
        return self._length_check(key, self.n, word_count=False)


class FirstNCharactersFirstTokenSorted(BlockingRule):
    def __init__(self, blocking_column: str, n: int = 3, remove_non_alphanumerical=False):
        super().__init__(blocking_column)
        self.n = n
        self.remove_non_alphanumerical = remove_non_alphanumerical

    def __repr__(self):
        return f"first_{self.n}_characters_first_token_sorted_{self.blocking_column}"

    def _blocking_rule(self, c):
        if self.remove_non_alphanumerical:
            c = F.regexp_replace(c, r'\W+', ' ')
        tokens = F.split(c, r'\s+')
        sorted_tokens = F.sort_array(tokens)
        filtered_tokens = F.filter(sorted_tokens, lambda x: F.length(x) >= self.n)
        first_token = filtered_tokens.getItem(0)
        return F.substring(first_token, 1, self.n)


class LastNChars(BlockingRule):
    def __init__(self, blocking_column: str, n: int = 3):
        super().__init__(blocking_column)
        self.n = n

    def __repr__(self):
        return f"last_{self.n}_characters_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        key = F.substring(c, -self.n, self.n)
        return self._length_check(key, self.n)


class WholeField(BlockingRule):
    def __init__(self, blocking_column: str):
        super().__init__(blocking_column)

    def __repr__(self):
        return f"whole_field_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        return c


class FirstNWords(BlockingRule):
    def __init__(self, blocking_column: str, n: int = 1, remove_non_alphanumerical=False):
        super().__init__(blocking_column)
        self.n = n
        self.remove_non_alphanumerical = remove_non_alphanumerical

    def __repr__(self):
        return f"first_{self.n}_words_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        if self.remove_non_alphanumerical:
            c = F.regexp_replace(c, r'\W+', ' ')
        tokens = F.split(c, r'\s+')
        key = self._length_check(tokens, self.n, word_count=True)
        return F.array_join(F.slice(key, 1, self.n), ' ')


class FirstNLettersNoSpace(BlockingRule):
    def __init__(self, blocking_column: str, n: int = 3):
        super().__init__(blocking_column)
        self.n = n

    def __repr__(self):
        return f"first_{self.n}_letters_{self.blocking_column}_no_space"

    def _blocking_rule(self, c: Column) -> Column:
        key = F.substring(F.regexp_replace(c, r'[^a-zA-Z]+', ''), 1, self.n)
        return self._length_check(key, self.n)


class SortedIntegers(BlockingRule):
    def __init__(self, blocking_column: str):
        super().__init__(blocking_column)

    def __repr__(self):
        return f"sorted_integers_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        number_string = F.trim(F.regexp_replace(c, r'[^0-9\s]+', ''))
        number_string_array = F.when(number_string != '', F.split(number_string, r'\s+'))
        number_int_array = F.transform(number_string_array, lambda x: x.cast(T.IntegerType()))
        number_sorted = F.array_sort(number_int_array)
        return F.array_join(number_sorted, " ")


class FirstInteger(BlockingRule):
    def __init__(self, blocking_column: str):
        super().__init__(blocking_column)

    def __repr__(self):
        return f"first_integer_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        number_string_array = F.split(F.trim(F.regexp_replace(c, r'[^0-9\s]+', '')), r'\s+')
        number_int_array = F.transform(number_string_array, lambda x: x.cast(T.IntegerType()))
        first_number = number_int_array.getItem(0)
        return first_number.cast(T.StringType())


class LastInteger(BlockingRule):
    def __init__(self, blocking_column: str):
        super().__init__(blocking_column)

    def __repr__(self):
        return f"last_integer_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        number_string_array = F.split(F.trim(F.regexp_replace(c, r'[^0-9\s]+', '')), r'\s+')
        number_int_array = F.transform(number_string_array, lambda x: x.cast(T.IntegerType()))
        last_number = F.slice(number_int_array, -1, 1).getItem(0)
        return last_number.cast(T.StringType())


class LargestInteger(BlockingRule):
    def __init__(self, blocking_column: str):
        super().__init__(blocking_column)

    def __repr__(self):
        return f"largest_integer_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        number_string_array = F.split(F.trim(F.regexp_replace(c, r'[^0-9\s]+', '')), r'\s+')
        number_int_array = F.transform(number_string_array, lambda x: x.cast(T.IntegerType()))
        largest_number = F.array_max(number_int_array)
        return largest_number.cast(T.StringType())


class NLetterAbbreviation(BlockingRule):
    def __init__(self, blocking_column: str, n: int = 3):
        super().__init__(blocking_column)
        self.n = n

    def __repr__(self):
        return f"{self.n}_letter_abbreviation_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        words = F.split(F.trim(F.regexp_replace(c, r'[0-9]+', '')), r'\s+')
        first_letters = F.when(F.size(words) >= self.n, F.transform(words, lambda x: F.substring(x, 1, 1)))
        return F.array_join(first_letters, '')


# this is an example of a blocking rule that contains a udf with plain python code:

class UdfFirstNChar(BlockingRule):
    def __init__(self, blocking_column: str, n: int):
        super().__init__(blocking_column)
        self.n = n

    def __repr__(self):
        return f"udf_first_integer_{self.blocking_column}"

    def _blocking_rule(self, c: Column) -> Column:
        @F.udf
        def _rule(s: str) -> str:
            return s[:self.n]

        return _rule(c)


default_blocking_rules = [FirstNChars, FirstNCharsLastWord, LastNChars, WholeField, FirstNWords, FirstNLettersNoSpace,
                          SortedIntegers, FirstInteger, LastInteger, LargestInteger, NLetterAbbreviation]
