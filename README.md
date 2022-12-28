<!--- BADGES: START --->
[![Version](https://img.shields.io/pypi/v/spark-matcher)](https://pypi.org/project/spark-matcher/)
[![Downloads](https://pepy.tech/badge/spark-matcher)](https://pepy.tech/project/spark-matcher)
![](https://img.shields.io/github/license/ing-bank/spark-matcher)
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=readthdocs&style=flat&color=pink&label=docs&message=spark-matcher)][#docs-package]

[#docs-package]: https://spark-matcher.readthedocs.io/en/latest/
<!--- BADGES: END --->

![spark_matcher_logo](https://spark-matcher.readthedocs.io/en/latest/_images/spark_matcher_logo.png)

# Spark-Matcher

Spark-Matcher is a scalable entity matching algorithm implemented in PySpark. With Spark-Matcher the user can easily
train an algorithm to solve a custom matching problem. Spark Matcher uses active learning (modAL) to train a
classifier (Scikit-learn) to match entities. In order to deal with the N^2 complexity of matching large tables, blocking is
implemented to reduce the number of pairs. Since the implementation is done in PySpark, Spark-Matcher can deal with
extremely large tables.

Documentation with examples can be found [here](https://spark-matcher.readthedocs.io/en/latest/).

Developed by data scientists at ING Analytics, www.ing.com.

## Installation

### Normal installation

As Spark-Matcher is intended to be used with large datasets on a Spark cluster, it is assumed that Spark is already 
installed. If that is not the case, first install Spark and PyArrow (`pip install pyspark pyarrow`).

Install Spark-Matcher using PyPi:

```
pip install spark-matcher
```

### Install with possibility to create documentation

Pandoc, the general markup converter needs to be available. You may follow the official [Pandoc installations instructions](https://pandoc.org/installing.html) or use conda:

```
conda install -c conda-forge pandoc
```

Then clone the Spark-Matcher repository and add `[doc]` like this:

```
pip install ".[doc]"
```

### Install to contribute

Clone this repo and install in editable mode. This also installs PySpark and Jupyterlab:

```
python -m pip install -e ".[dev]"
python setup.py develop
```

## Documentation

Documentation can be created using the following command:

```
make create_documentation
```

## Dependencies

The usage examples in the `examples` directory contain notebooks that run in local mode. 
Using the SparkMatcher in cluster mode, requires sending the SparkMatcher package and several other python packages (see spark_requirements.txt) to the executors.
How to send these dependencies, depends on the cluster. 
Please read the instructions and examples of Apache Spark on how to do this: https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html.

SparkMatcher uses `graphframes` under to hood. 
Therefore, depending on the spark version, the correct version of `graphframes` needs to be added to the `external_dependencies` directory and to the configuration of the spark session.  
As a default, `graphframes` for spark 3.0 is used in the spark sessions in the notebooks in the `examples` directory. 
For a different version, see: https://spark-packages.org/package/graphframes/graphframes.

## Usage

Example notebooks are provided in the `examples` directory.
Using the SparkMatcher to find matches between Spark
dataframes `a` and `b` goes as follows:

```python
from spark_matcher.matcher import Matching

myMatcher = Matcher(spark_session, col_names=['name', 'suburb', 'postcode'])
```

Now we are ready for fitting the Matcher object using 'active learning'; this means that the user has to enter whether a
pair is a match or not. You enter 'y' if a pair is a match or 'n' when a pair is not a match. You will be notified when
the model has converged and you can stop training by pressing 'f'.

```python
myMatcher.fit(a, b)
```

The Matcher is now trained and can be used to predict on all data. This can be the data used for training or new data
that was not seen by the model yet.

```python
result = myMatcher.predict(a, b)
```
