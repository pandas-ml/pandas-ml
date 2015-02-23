
.. ipython:: python
   :suppress:

   from pandas import options
   options.display.max_rows = 15
   options.display.max_columns = 7


Use scikit-learn
================

This section describes how to use ``scikit-learn`` functionalities via ``expandas``.

Basics
------

You can create ``ModelFrame`` instance from ``scikit-learn`` datasets directly.

.. ipython:: python

   import expandas as expd
   import sklearn.datasets as datasets

   df = expd.ModelFrame(datasets.load_iris())
   df.head()


ModelFrame has accessor methods which makes easier access to ``scikit-learn`` namespace.

.. ipython:: python

   df.cluster.KMeans

Thus, you can instanciate each estimator via ``ModelFrame`` accessors. Once create an estimator, you can pass it to ``ModelFrame.fit`` then ``predict``. `ModelFrame`` automatically uses its data and target properties for each operations.

.. ipython:: python

   estimator = df.cluster.KMeans(n_clusters=3)
   df.fit(estimator)
   predicted = df.predict(estimator)
   predicted

``ModelFrame`` has following methods corresponding to various ``scikit-learn`` estimators.

- ``ModelFrame.fit``
- ``ModelFrame.transform``
- ``ModelFrame.fit_transform``
- ``ModelFrame.inverse_transform``
- ``ModelFrame.predict``
- ``ModelFrame.fit_predict``
- ``ModelFrame.score``

Following example shows to perform PCA.

.. ipython:: python

   estimator = df.decomposition.PCA()
   df.fit(estimator)
   transformed = df.transform(estimator)
   transformed.head()
   type(transformed)

   transformed.inverse_transform(estimator)

.. note:: ``columns`` information will be lost once transformed to principal components.

Cross Validation
----------------

``scikit-learn`` has some classes for cross validation, and these classes returns indexes for training sets and test sets. You can slice ``ModelFrame using these indexes.

.. ipython:: python

   kf = df.cross_validation.KFold(n=150, n_folds=3)
   for train_index, test_index in kf:
      print('training set shape: ', df.iloc[train_index, :].shape,
            'test set shape: ', df.iloc[test_index, :].shape)


For further simplification, ``ModelFrame.cross_validation.iterate`` can accept such iterators and returns ``ModelFrame`` corresponding to training and test data.

.. ipython:: python

   kf = df.cross_validation.KFold(n=150, n_folds=3)
   for train_df, test_df in df.cross_validation.iterate(kf):
      print('training set shape: ', train_df.shape,
            'test set shape: ', test_df.shape)

Grid Search
-----------


