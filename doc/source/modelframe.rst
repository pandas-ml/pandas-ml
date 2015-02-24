
Data Handling
=============

Data Preparation
----------------

This section describes how to prepare basic data format named ``ModelFrame``. ``ModelFrame`` defines a metadata to specify target (response variable) and data (explanatory variable / features). Using these metadata, ``ModelFrame`` can call other statistics/ML functions in more simple way.

You can create ``ModelFrame`` as the same manner as ``pandas.DataFrame``. The below example shows how to create basic ``ModelFrame``, which DOESN'T have target values.

.. code-block:: python

   >>> import expandas as expd

   >>> df = expd.ModelFrame({'A': [1, 2, 3], 'B': [2, 3, 4],
   ...                       'C': [3, 4, 5]}, index=['A', 'B', 'C'])
   >>> df
      A  B  C
   A  1  2  3
   B  2  3  4
   C  3  4  5

   >>> type(df)
   <class 'expandas.core.frame.ModelFrame'>


You can check whether the created ``ModelFrame`` has target values using ``ModelFrame.has_target()`` function.

.. code-block:: python

   >>> df.has_target()
   False

Target values can be specifyied via ``target`` keyword. You can simply pass a column name to be handled as target. Target column name can be confirmed via ``target_name`` property.

.. code-block:: python

   >>> df2 = expd.ModelFrame({'A': [1, 2, 3], 'B': [2, 3, 4],
   ...                        'C': [3, 4, 5]}, target='A')
   >>> df2
      A  B  C
   0  1  2  3
   1  2  3  4
   2  3  4  5

   >>> df2.has_target()
   True

   >>> df2.target_name
   'A'

Also, you can pass any list-likes to be handled as a target. In this case, target column will be named as ".target".

.. code-block:: python

   >>> df3 = expd.ModelFrame({'A': [1, 2, 3], 'B': [2, 3, 4],
   ...                        'C': [3, 4, 5]}, target=[4, 5, 6])
   >>> df3
      .target  A  B  C
   0        4  1  2  3
   1        5  2  3  4
   2        6  3  4  5

   >>> df3.has_target()
   True

   >>> df3.target_name
   '.target'

Also, you can pass ``pandas.DataFrame`` and ``pandas.Series`` as data and target.

.. code-block:: python

   >>> import pandas as pd
   df4 = expd.ModelFrame({'A': [1, 2, 3], 'B': [2, 3, 4],
   ...                    'C': [3, 4, 5]}, target=pd.Series([4, 5, 6]))
   >>> df4
      .target  A  B  C
   0        4  1  2  3
   1        5  2  3  4
   2        6  3  4  5

   >>> df4.has_target()
   True

   >>> df4.target_name
   '.target'

.. note:: Target values are mandatory to perform operations which require response variable, such as regression and supervised learning.


Data Manipulation
-----------------

You can access to each property as the same as ``pandas.DataFrame``. Sliced results will be ``ModelSeries`` (simple wrapper for ``pandas.Series`` to support some data manipulation) or ``ModelFrame``

.. code-block:: python

   >>> df
      A  B  C
   A  1  2  3
   B  2  3  4
   C  3  4  5

   >>> sliced = df['A']
   >>> sliced
   A    1
   B    2
   C    3
   Name: A, dtype: int64

   >>> type(sliced)
   <class 'expandas.core.series.ModelSeries'>

   >>> subset = df[['A', 'B']]
   >>> subset
      A  B
   A  1  2
   B  2  3
   C  3  4

   >>> type(subset)
   <class 'expandas.core.frame.ModelFrame'>

``ModelFrame`` has a special properties to access data (features) an target.

.. code-block:: python

   >>> df2
      A  B  C
   0  1  2  3
   1  2  3  4
   2  3  4  5

   >>> df2.target_name
   'A'

   >>> df2.data
      B  C
   0  2  3
   1  3  4
   2  4  5

   >>> df2.target
   0    1
   1    2
   2    3
   Name: A, dtype: int64


You can update data and target via properties, in addition to standard ``pandas.DataFrame`` ways.

.. code-block:: python

   >>> df2.target = [9, 9, 9]
   >>> df2
      A  B  C
   0  9  2  3
   1  9  3  4
   2  9  4  5

   >>> df2.data = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
   >>> df2
      A  X  Y
   0  9  1  4
   1  9  2  5
   2  9  3  6

   >>> df2['X'] = [0, 0, 0]
   >>> df2
      A  X  Y
   0  9  0  4
   1  9  0  5
   2  9  0  6

You can change target column specifying ``target_name`` property. Specifying a column which doesn't exist in ``ModelFrame`` results in target column to be data column.

.. code-block:: python

   >>> df2.target_name
   'A'

   >>> df2.target_name = 'X'
   >>> df2.target_name
   'X'

   >>> df2.target_name = 'XXXX'
   >>> df2.has_target()
   False

   >>> df2.data
      A  X  Y
   0  9  0  4
   1  9  0  5
   2  9  0  6
