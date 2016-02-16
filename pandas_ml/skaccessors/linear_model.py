#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods, _attach_methods, _wrap_data_target_func


class LinearModelMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.linear_model``.
    """

    _module_name = 'sklearn.linear_model'

    def enet_path(self, *args, **kwargs):
        """
        Call ``sklearn.linear_model.enet_path`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.enet_path
        return self._enet_path_wraps(func, *args, **kwargs)

    def _enet_path_wraps(self, func, *args, **kwargs):
        data = self._data
        target = self._target
        return_models = kwargs.get('return_models', False)
        if return_models:
            models = func(data.values, y=target.values, *args, **kwargs)
            return models
        else:
            alphas, coefs, dual_gaps = func(data.values, y=target.values, *args, **kwargs)
            coefs = self._constructor(coefs, index=data.columns)
            return alphas, coefs, dual_gaps

    def lars_path(self, *args, **kwargs):
        """
        Call ``sklearn.linear_model.lars_path`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.lars_path
        data = self._data
        target = self._target
        alphas, active, coefs = func(data.values, y=target.values, *args, **kwargs)
        coefs = self._constructor(coefs, index=data.columns)
        return alphas, active, coefs

    def lasso_path(self, *args, **kwargs):
        """
        Call ``sklearn.linear_model.lasso_path`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.lasso_path
        # lasso_path internally uses enet_path
        return self._enet_path_wraps(func, *args, **kwargs)

    def lasso_stability_path(self, *args, **kwargs):
        """
        Call ``sklearn.linear_model.lasso_stability_path`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.lasso_stability_path
        data = self._data
        target = self._target
        alpha_grid, scores_path = func(data.values, y=target.values, *args, **kwargs)
        scores_path = self._constructor(scores_path, index=data.columns)
        return alpha_grid, scores_path

    def orthogonal_mp_gram(self, *args, **kwargs):
        """
        Call ``sklearn.linear_model.orthogonal_mp_gram`` using automatic mapping.

        - ``Gram``: ``ModelFrame.data.T.dot(ModelFrame.data)``
        - ``Xy``: ``ModelFrame.data.T.dot(ModelFrame.target)``
        """
        func = self._module.orthogonal_mp_gram
        data = self._data.values
        target = self._target.values
        gram = data.T.dot(data)
        Xy = data.T.dot(target)
        coef = func(gram, Xy, *args, **kwargs)
        return coef


_lm_methods = ['orthogonal_mp']
_attach_methods(LinearModelMethods, _wrap_data_target_func, _lm_methods)
