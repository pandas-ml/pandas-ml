#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


class CovarianceMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.covariance``.
    """

    _module_name = 'sklearn.covariance'

    def empirical_covariance(self, *args, **kwargs):
        """
        Call ``sklearn.covariance.empirical_covariance`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.empirical_covariance
        data = self._data
        covariance = func(data.values, *args, **kwargs)
        covariance = self._constructor(covariance, index=data.columns, columns=data.columns)
        return covariance

    def ledoit_wolf(self, *args, **kwargs):
        """
        Call ``sklearn.covariance.ledoit_wolf`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.ledoit_wolf
        data = self._data
        shrunk_cov, shrinkage = func(data.values, *args, **kwargs)
        shrunk_cov = self._constructor(shrunk_cov, index=data.columns, columns=data.columns)
        return shrunk_cov, shrinkage

    def oas(self, *args, **kwargs):
        """
        Call ``sklearn.covariance.oas`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.oas
        data = self._data
        shrunk_cov, shrinkage = func(data.values, *args, **kwargs)
        shrunk_cov = self._constructor(shrunk_cov, index=data.columns, columns=data.columns)
        return shrunk_cov, shrinkage
