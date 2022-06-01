.. Spark Matcher documentation master file, created by
   sphinx-quickstart on Tue Nov 23 10:39:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/spark_matcher_logo.png

Welcome to Spark Matcher's documentation!
=========================================

Spark Matcher is a scalable entity matching algorithm implemented in PySpark.
With Spark Matcher the user can easily train an algorithm to solve a custom matching problem.
Spark Matcher uses active learning (modAL) to train a classifier (Sklearn) to match entities.
In order to deal with the N^2 complexity of matching large tables, blocking is implemented to reduce the number of pairs.
Since the implementation is done in PySpark, Spark Matcher can deal with extremely large tables.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation_guide
   api/modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
