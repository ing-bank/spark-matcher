import numpy as np
import pandas as pd


def test_parquetcheckpointer(spark_session):
    from spark_matcher.table_checkpointer import ParquetCheckPointer

    checkpointer = ParquetCheckPointer(spark_session, 'temp_database', 'checkpoint_name')

    pdf = pd.DataFrame({'col_1': ['1', '2', '3'],
                        'col_2': ['a', 'b', 'c'],
                        })
    sdf = spark_session.createDataFrame(pdf)

    returned_sdf = checkpointer(sdf, 'checkpoint_name')
    returned_pdf = returned_sdf.toPandas().sort_values(['col_1', 'col_2'])
    np.array_equal(pdf.values, returned_pdf.values)
