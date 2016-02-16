#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


class DecompositionMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.decomposition``.
    """

    _module_name = 'sklearn.decomposition'

    def fastica(self, *args, **kwargs):
        """
        Call ``sklearn.decomposition.fastica`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.fastica
        data = self._data
        return_x_mean = kwargs.get('return_X_mean', False)

        if return_x_mean:
            K, W, S, X_mean = func(data.values, *args, **kwargs)
            K = self._constructor(K, index=data.columns)
            W = self._constructor(W)
            S = self._constructor(S, index=data.index)
            return K, W, S, X_mean
        else:
            K, W, S = func(data.values, *args, **kwargs)
            K = self._constructor(K, index=data.columns)
            W = self._constructor(W)
            S = self._constructor(S, index=data.index)
            return K, W, S

    def dict_learning(self, n_components, alpha, *args, **kwargs):
        """
        Call ``sklearn.decomposition.dict_learning`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.dict_learning
        data = self._data
        code, dictionary, errors = func(data.values, n_components, alpha, *args, **kwargs)
        code = self._constructor(code, index=data.index)
        dictionary = self._constructor(dictionary, columns=data.columns)
        return code, dictionary, errors

    def dict_learning_online(self, *args, **kwargs):
        """
        Call ``sklearn.decomposition.dict_learning_online`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.dict_learning_online
        data = self._data
        return_code = kwargs.get('return_code', True)
        if return_code:
            code, dictionary = func(data.values, *args, **kwargs)
            code = self._constructor(code, index=data.index)
            dictionary = self._constructor(dictionary, columns=data.columns)
            return code, dictionary
        else:
            dictionary = func(data.values, *args, **kwargs)
            dictionary = self._constructor(dictionary, columns=data.columns)
            return dictionary

    def sparse_encode(self, dictionary, *args, **kwargs):
        """
        Call ``sklearn.decomposition.sparce_encode`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.sparse_encode
        data = self._data
        code = func(data.values, dictionary, *args, **kwargs)
        code = self._constructor(code, index=data.index)
        return code
