
What's new
==========

v0.2.0
------

Enhancement
^^^^^^^^^^^

- ``ModelFrame.transform`` can preserve column names for some ``sklearn.preprocessing`` transformation.
- Added ``ModelSeries.fit``, ``transform``, ``fit_transform`` and ``inverse_transform`` for preprocessing purpose.
- ``ModelFrame`` can be initialized from ``statsmodels`` datasets.

Bug Fix
^^^^^^^

- ``target`` kw may be ignored when initializing ``ModelFrame`` with ``np.ndarray`` and ``columns`` kwds.

v0.1.1
------

Enhancement
^^^^^^^^^^^

- Added ``sklearn.learning_curve``, ``neural_network``, ``random_projection``

v0.1.0
------

- Initial Release