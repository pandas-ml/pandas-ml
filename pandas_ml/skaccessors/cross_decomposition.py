#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


class CrossDecompositionMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.cross_decomposition``.
    """

    _module_name = 'sklearn.cross_decomposition'
    _method_mapper = dict(fit={'PLSCanonical': '_fit', 'CCA': '_fit', 'PLSRegression': '_fit'},
                          transform={'PLSCanonical': '_transform', 'CCA': '_transform'},
                          predict={'PLSRegression': '_predict'})

    @classmethod
    def _fit(cls, df, estimator, *args, **kwargs):
        data = df.data.values
        if df.has_target():
            target = df.target.values
            result = estimator.fit(data, Y=target, *args, **kwargs)
        else:
            # not try to pass target if it doesn't exists
            # to catch ValueError from estimator
            result = estimator.fit(data, *args, **kwargs)
        return result

    @classmethod
    def _transform(cls, df, estimator, *args, **kwargs):
        data = df.data.values
        if df.has_target():
            target = df.target.values
            try:
                result = estimator.transform(data, Y=target, *args, **kwargs)
                result = df._constructor(result[0], target=result[1])
            except TypeError:
                result = estimator.transform(data, *args, **kwargs)
                result = df._constructor(result)
        else:
            # not try to pass target if it doesn't exists
            # to catch ValueError from estimator
            result = estimator.transform(data, *args, **kwargs)
            result = df._constructor(result)
        return result

    @classmethod
    def _predict(cls, df, estimator, *args, **kwargs):
        data = df.data.values
        result = estimator.predict(data, *args, **kwargs)
        result = df._constructor(result)
        return result
