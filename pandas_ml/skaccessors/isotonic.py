#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


class IsotonicMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.isotonic``.
    """

    _module_name = 'sklearn.isotonic'

    @property
    def IsotonicRegression(self):
        """``sklearn.isotonic.IsotonicRegression``"""
        return self._module.IsotonicRegression

    def isotonic_regression(self, *args, **kwargs):
        """
        Call ``sklearn.isotonic.isotonic_regression`` using automatic mapping.

        - ``y``: ``ModelFrame.target``
        """
        func = self._module.isotonic_regression
        target = self._target
        _y = func(target.values, *args, **kwargs)
        _y = self._constructor_sliced(_y, index=target.index)
        return _y

    def check_increasing(self, *args, **kwargs):
        """
        Call ``sklearn.isotonic.check_increasing`` using automatic mapping.

        - ``x``: ``ModelFrame.index``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.check_increasing
        target = self._target
        return func(target.index, target.values, *args, **kwargs)
