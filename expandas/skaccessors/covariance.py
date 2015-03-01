#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class CovarianceMethods(AccessorMethods):
    """
    Accessor to ``sklearn.covariance``.
    """

    _module_name = 'sklearn.covariance'

    def empirical_covariance(self, *args, **kwargs):
        func = self._module.empirical_covariance
        data = self.data
        covariance = func(data.values, *args, **kwargs)
        covariance = self._constructor(covariance, index=data.columns, columns=data.columns)
        return covariance

    def ledoit_wolf(self, *args, **kwargs):
        func = self._module.ledoit_wolf
        data = self.data
        shrunk_cov, shrinkage = func(data.values, *args, **kwargs)
        shrunk_cov = self._constructor(shrunk_cov, index=data.columns, columns=data.columns)
        return shrunk_cov, shrinkage

    def oas(self, *args, **kwargs):
        func = self._module.oas
        data = self.data
        shrunk_cov, shrinkage = func(data.values, *args, **kwargs)
        shrunk_cov = self._constructor(shrunk_cov, index=data.columns, columns=data.columns)
        return shrunk_cov, shrinkage
