from setuptools import setup, find_packages

base_packages = [
    'pandas',
    'numpy',
    'scikit-learn',
    'python-Levenshtein',
    'thefuzz',
    'modAL',
    'pytest',
    'multipledispatch',
    'dill',
    'graphframes',
    'scipy'
]

doc_packages = [
    'sphinx',
    'nbsphinx',
    'sphinx_rtd_theme'
]

util_packages = [
    'pyspark',
    'pyarrow',
    'jupyterlab'
]

base_doc_packages = base_packages + doc_packages
dev_packages = base_packages + doc_packages + util_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='Spark-Matcher',
      version='0.2.4',
      author="Ahmet Bayraktar, Stan Leisink, Frits Hermans",
      description="Record matching and entity resolution at scale in Spark",
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      packages=find_packages(exclude=['examples']),
      package_data={"spark_matcher": ["data/*.csv"]},
      install_requires=base_packages,
      extras_require={
          "base": base_packages,
          "doc": base_doc_packages,
          "dev": dev_packages,
      },
      python_requires=">=3.7",
      )
