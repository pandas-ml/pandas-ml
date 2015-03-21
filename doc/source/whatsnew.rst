
What's new
==========

v0.2.0
------

Enhancement
^^^^^^^^^^^

- ``ModelFrame.transform`` can preserve column names for some ``sklearn.preprocessing`` transformation.
- Added ``ModelSeries.fit``, ``transform``, ``fit_transform`` and ``inverse_transform`` for preprocessing purpose.
- ``ModelFrame`` can be initialized from ``statsmodels`` datasets.
- ``ModelFrame.cross_validation.iterate`` and ``ModelFrame.cross_validation.train_test_split`` now keep index of original dataset, and added ``reset_index`` keyword to control this behaviour.

Bug Fix
^^^^^^^

- ``target`` kw may be ignored when initializing ``ModelFrame`` with ``np.ndarray`` and ``columns`` kwds.
- ``linear_model.enet_path`` doesn't accept additional keywords.


v0.1.1
------

Enhancement
^^^^^^^^^^^

- Added ``sklearn.learning_curve``, ``neural_network``, ``random_projection``

v0.1.0
------

- Initial Release