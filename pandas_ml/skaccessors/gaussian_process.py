#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods, _attach_methods, _wrap_data_func


class GaussianProcessMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.gaussian_process``.
    """

    _module_name = 'sklearn.gaussian_process'
    _method_mapper = dict(predict={'GaussianProcess': '_predict'})

    @property
    def correlation_models(self):
        """Property to access ``sklearn.gaussian_process.correlation_models``"""

        module_name = 'sklearn.gaussian_process.correlation_models'
        attrs = ['absolute_exponential', 'squared_exponential',
                 'generalized_exponential', 'pure_nugget',
                 'cubic', 'linear']
        return _AccessorMethods(self._df, module_name=module_name, attrs=attrs)

    @property
    def regression_models(self):
        """Property to access ``sklearn.gaussian_process.regression_models``"""

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


class RegressionModelsMethods(_AccessorMethods):
    _module_name = 'sklearn.gaussian_process.regression_models'


_regression_methods = ['constant', 'linear', 'quadratic']
_attach_methods(RegressionModelsMethods, _wrap_data_func, _regression_methods)
