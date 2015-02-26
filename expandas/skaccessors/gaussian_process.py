#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.core.accessor import AccessorMethods, _attach_methods


class GaussianProcessMethods(AccessorMethods):
    _module_name = 'sklearn.gaussian_process'

    @property
    def correlation_models(self):
        return CorrelationModelsMethods(self._df)

    @property
    def regression_models(self):
        return RegressionModelsMethods(self._df)


class CorrelationModelsMethods(AccessorMethods):
    _module_name = 'sklearn.gaussian_process.correlation_models'

    @property
    def absolute_exponential(self):
        return self._module.absolute_exponential

    @property
    def squared_exponential(self):
        return self._module.squared_exponential

    @property
    def generalized_exponential(self):
        return self._module.generalized_exponential

    @property
    def pure_nugget(self):
        return self._module.pure_nugget

    @property
    def cubic(self):
        return self._module.cubic

    @property
    def linear(self):
        return self._module.linear


class RegressionModelsMethods(AccessorMethods):
    _module_name = 'sklearn.gaussian_process.regression_models'


_regression_methods = ['constant', 'linear', 'quadratic']


def _wrap_func(func):
    def f(self, *args, **kwargs):
        data = self.data
        result = func(data.values, *args, **kwargs)
        return result
    return f


# _attach_methods(CorrelationModelsMethods, lambda f: f, _correlation_methods)
_attach_methods(RegressionModelsMethods, _wrap_func, _regression_methods)

