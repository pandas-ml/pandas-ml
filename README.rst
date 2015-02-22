expandas
========

.. image:: https://readthedocs.org/projects/expandas/badge/?version=latest
    :target: http://expandas.readthedocs.org/en/latest/
    :alt: Latest Docs

.. image:: https://travis-ci.org/sinhrks/expandas.svg?branch=master
    :target: https://travis-ci.org/sinhrks/expandas

Overview
~~~~~~~~

Expand pandas functionalities to be easier to use other statistics/ML packages, mainly forcusing on scikit-learn.

.. code-block:: python

    >>> import expandas as expd
    >>> import sklearn.datasets as datasets

    # create ModelFrame instance from sklearn.datasets
    >>> df = expd.ModelFrame(datasets.load_digits())
    >>> type(df)
    <class 'expandas.core.frame.ModelFrame'>

    # binarize data (features), not touching target
    >>> df.data = df.data.preprocessing.binarize()
    >>> df.head()
       .target  0  1  2  3  4  5  6  7  8 ...  54  55  56  57  58  59  60  61  62  63
    0        0  0  0  1  1  1  1  0  0  0 ...   0   0   0   0   1   1   1   0   0   0
    1        1  0  0  0  1  1  1  0  0  0 ...   0   0   0   0   0   1   1   1   0   0
    2        2  0  0  0  1  1  1  0  0  0 ...   1   0   0   0   0   1   1   1   1   0
    3        3  0  0  1  1  1  1  0  0  0 ...   1   0   0   0   1   1   1   1   0   0
    4        4  0  0  0  1  1  0  0  0  0 ...   0   0   0   0   0   1   1   1   0   0
    [5 rows x 65 columns]

    # split to training and test data
    >>> train_df, test_df = df.cross_validation.train_test_split()

    # create estimator (accessor is mapped to sklearn namespace)
    >>> estimator = df.svm.LinearSVC(C=1.0)

    # fit to training data
    >>> train_df.fit(estimator)

    # predict test data
    >>> test_df.predict(estimator)
    0     4
    1     2
    2     7
    ...
    448    5
    449    8
    Length: 450, dtype: int64

    # Evaluate the result
    >>> test_df.metrics.confusion_matrix()
        0   1   2   3   4   5   6   7   8   9
    0  38   1   0   0   0   0   0   0   0   0
    1   0  34   1   2   1   0   0   0   2   5
    2   0   1  37   1   0   0   0   1   2   0
    3   0   2   3  38   0   0   0   0   2   2
    4   0   0   0   0  47   0   0   0   0   1
    5   0   0   0   0   0  34   0   0   0   3
    6   0   2   0   0   1   1  45   0   0   0
    7   0   1   0   0   2   0   0  38   0   1
    8   0   2   0   3   0   1   1   0  42   0
    9   0   2   0   4   1   1   0   0   1  43


Supported Packages
~~~~~~~~~~~~~~~~~~

**IMPORTANT**: All the implementations are highly experimental, and forced to be changed without notice.

- scikit-learn
    - cluster
    - cross_validation
    - decomposition
    - dummy
    - ensemble
    - lda
    - linear_model
    - feature_selection
    - naive_bayes
    - neighbors
    - metrics (partially)
    - preprocessing (not well tested)
    - svm
    - tree

