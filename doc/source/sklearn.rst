
.. ipython:: python
   :suppress:

   from pandas import options
   options.display.max_rows = 10
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

   # make readable
   df.columns = ['.target', 'sepal length', 'sepal width', 'petal length', 'petal width']


``ModelFrame`` has accessor methods which makes easier access to ``scikit-learn`` namespace.

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

``ModelFrame`` preserves last predicted result, and perform evaluation using functions available in ``ModelFrame.metrics``.

.. ipython:: python

   estimator = df.svm.SVC()
   df.fit(estimator)
   df.predict(estimator)
   df.metrics.confusion_matrix()

Pipeline
--------

``ModelFrame`` can handle pipeline as the same as normal estimators.

.. ipython:: python

   estimators = [('reduce_dim', df.decomposition.PCA()),
                 ('svm', df.svm.SVC())]
   pipe = df.pipeline.Pipeline(estimators)

   df.fit(pipe)
   df.predict(pipe)

Cross Validation
----------------

``scikit-learn`` has some classes for cross validation. The most easiest way is to use ``train_test_split`` to split data to training and test set. You can access to the function via ``cross_validation`` accessor.

.. ipython:: python

   df
   train_df, test_df = df.cross_validation.train_test_split()

   train_df
   test_df


Also, there are some iterative classes which returns indexes for training sets and test sets. You can slice ``ModelFrame`` using these indexes.

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

You can perform grid search using ``ModelFrame.fit``.

.. ipython:: python

   tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100]},
                       {'kernel': ['linear'], 'C': [1, 10, 100]}]

   df = expd.ModelFrame(datasets.load_digits())
   cv = df.grid_search.GridSearchCV(df.svm.SVC(C=1), tuned_parameters,
                                    cv=5, scoring='precision')
   df.fit(cv)
   cv.best_estimator_

In addition, ``ModelFrame.grid_search`` has a ``describe`` function to organize each grid search result as ``pd.DataFrame`` accepting estimator.

.. ipython:: python

   df.grid_search.describe(cv)



