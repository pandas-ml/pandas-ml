
Use patsy
=========

This section describes data transformation using ``patsy``. ``ModelFrame.transform`` can accept ``patsy`` style formula.

.. code-block:: python

   >>> import pandas_ml as pdml

   # create modelframe which doesn't have target
   >>> df = pdml.ModelFrame({'X': [1, 2, 3], 'Y': [2, 3, 4],
   ...                       'Z': [3, 4, 5]}, index=['a', 'b', 'c'])

   >>> df
      X  Y  Z
   a  1  2  3
   b  2  3  4
   c  3  4  5

   # transform with patsy formula
   >>> transformed = df.transform('Z ~ Y + X')
   >>> transformed
      Z  Intercept  Y  X
   a  3          1  2  1
   b  4          1  3  2
   c  5          1  4  3

   # transformed data should have target specified by formula
   >>> transformed.target
   a    3
   b    4
   c    5
   Name: Z, dtype: float64

   >>> transformed.data
      Intercept  Y  X
   a          1  2  1
   b          1  3  2
   c          1  4  3


If you do not want intercept, specify with ``0``.

.. code-block:: python

   >>> df.transform('Z ~ Y + 0')
      Z  Y
   a  3  2
   b  4  3
   c  5  4


Also, you can use formula which doesn't have left side.

.. code-block:: python

   # create modelframe which has target
   >>> df2 = pdml.ModelFrame({'X': [1, 2, 3], 'Y': [2, 3, 4],'Z': [3, 4, 5]},
   ...                       target =[7, 8, 9], index=['a', 'b', 'c'])

   >>> df2
      .target  X  Y  Z
   a        7  1  2  3
   b        8  2  3  4
   c        9  3  4  5

   # overwrite data with transformed data
   >>> df2.data = df2.transform('Y + Z')
   >>> df2
      .target  Intercept  Y  Z
   a        7          1  2  3
   b        8          1  3  4
   c        9          1  4  5

   # data has been updated based on formula
   >>> df2.data
      Intercept  Y  Z
   a          1  2  3
   b          1  3  4
   c          1  4  5

   # target is not changed
   >>> df2.target
   a    7
   b    8
   c    9
   Name: .target, dtype: int64

Below example is performing deviation coding via patsy formula.

   >>> df3 = pdml.ModelFrame({'X': [1, 2, 3, 4, 5], 'Y': [1, 3, 2, 2, 1],
   ...                        'Z': [1, 1, 1, 2, 2]}, target='Z',
   ...                        index=['a', 'b', 'c', 'd', 'e'])

   >>> df3
      X  Y  Z
   a  1  1  1
   b  2  3  1
   c  3  2  1
   d  4  2  2
   e  5  1  2

   >>> df3.transform('C(X, Sum)')
      Intercept  C(X, Sum)[S.1]  C(X, Sum)[S.2]  C(X, Sum)[S.3]  C(X, Sum)[S.4]
   a          1               1               0               0               0
   b          1               0               1               0               0
   c          1               0               0               1               0
   d          1               0               0               0               1
   e          1              -1              -1              -1              -1

   >>> df3.transform('C(Y, Sum)')
      Intercept  C(Y, Sum)[S.1]  C(Y, Sum)[S.2]
   a          1               1               0
   b          1              -1              -1
   c          1               0               1
   d          1               0               1
   e          1               1               0