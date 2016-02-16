#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


class SVMMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.svm``.
    """

    _module_name = 'sklearn.svm'

    def l1_min_c(self, *args, **kwargs):
        """
        Call ``sklearn.svm.l1_min_c`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.l1_min_c
        data = self._data
        target = self._target
        l1_min_c = func(data.values, y=target.values, *args, **kwargs)
        return l1_min_c

    @property
    def libsvm(self):
        """Not implemented"""
        raise NotImplementedError

    @property
    def liblinear(self):
        """Not implemented"""
        raise NotImplementedError

    @property
    def libsvm_sparse(self):
        """Not implemented"""
        raise NotImplementedError
