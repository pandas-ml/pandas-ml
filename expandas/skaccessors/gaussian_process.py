#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.core.accessor import AccessorMethods, _attach_methods


class GaussianProcessMethods(AccessorMethods):
    _module_name = 'sklearn.gaussian_process'

    @property
    def correlation_models(self):
        module_name = 'sklearn.gaussian_process.correlation_models'
        attrs = ['absolute_exponential', 'squared_exponential',
                 'generalized_exponential', 'pure_nugget',
                 'cubic', 'linear']
        return AccessorMethods(self._df, module_name=module_name, attrs=attrs)

    @property
    def regression_models(self):
        return RegressionModelsMethods(self._df)

    @classmethod
    def _predict(cls, df, estimator, *args, **kwargs):
        data = df.data.values
        eval_MSE = kwargs.get('eval_MSE', False)
        if eval_MSE:
            y, MSE = estimator.predict(data, *args, **kwargs)
            if y.ndim == 1:
                y = df._constructor_sliced(y, index=df.index)
                MSE = df._constructor_sliced(MSE, index=df.index)
            else:
                y = df._constructor(y, index=df.index)
                MSE = df._constructor(MSE, index=df.index)
            return y, MSE
        else:
            y = estimator.predict(data, *args, **kwargs)
            if y.ndim == 1:
                y = df._constructor_sliced(y, index=df.index)
            else:
                y = df._constructor(y, index=df.index)
            return y


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

