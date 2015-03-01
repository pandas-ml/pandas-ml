#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class LinearModelMethods(AccessorMethods):
    """
    Accessor to ``sklearn.linear_model``.
    """

    _module_name = 'sklearn.linear_model'

    def lars_path(self, *args, **kwargs):
        func = self._module.lars_path
        data = self.data
        target = self.target
        alphas, active, coefs = func(data.values, y=target.values, *args, **kwargs)
        coefs = self._constructor(coefs, index=data.columns)
        return alphas, active, coefs

    def lasso_path(self, *args, **kwargs):
        func = self._module.lasso_path
        data = self.data
        target = self.target
        return_models = kwargs.get('return_models', False)
        if return_models:
            models = func(data.values, y=target.values, *args, **kwargs)
            return models
        else:
            alphas, coefs, dual_gaps = func(data.values, y=target.values, *args, **kwargs)
            coefs = self._constructor(coefs, index=data.columns)
            return alphas, coefs, dual_gaps

    def lasso_stability_path(self, *args, **kwargs):
        func = self._module.lasso_stability_path
        data = self.data
        target = self.target
        alpha_grid, scores_path = func(data.values, y=target.values, *args, **kwargs)
        scores_path = self._constructor(scores_path, index=data.columns)
        return alpha_grid, scores_path

    def orthogonal_mp_gram(self, *args, **kwargs):
        func = self._module.orthogonal_mp_gram
        data = self.data.values
        target = self.target.values
        gram = data.T.dot(data)
        Xy = data.T.dot(target)
        coef = func(gram, Xy, *args, **kwargs)
        return coef


_lm_methods = ['orthogonal_mp']


def _wrap_func(func):
    def f(self, *args, **kwargs):
        data = self.data
        target = self.target
        result = func(data.values, y=target.values, *args, **kwargs)
        return result
    return f


_attach_methods(LinearModelMethods, _wrap_func, _lm_methods)


