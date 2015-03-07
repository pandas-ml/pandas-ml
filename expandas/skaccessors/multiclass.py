#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.core.accessor import AccessorMethods


class MultiClassMethods(AccessorMethods):
    """
    Accessor to ``sklearn.multiclass``.
    """

    _module_name = 'sklearn.multiclass'

    @property
    def OneVsRestClassifier(self):
        return self._module.OneVsRestClassifier

    @property
    def OneVsOneClassifier(self):
        return self._module.OneVsOneClassifier

    @property
    def OutputCodeClassifier(self):
        return self._module.OutputCodeClassifier

    def fit_ovr(self, *args, **kwargs):
        """Deprecated"""
        msg = "sklearn.multiclass.fit_ovr is deprecated"
        raise NotImplementedError(msg)

    def predict_ovr(self, *args, **kwargs):
        """Deprecated"""
        msg = "sklearn.multiclass.predict_ovr is deprecated"
        raise NotImplementedError(msg)

    def fit_ovo(self, *args, **kwargs):
        """Deprecated"""
        msg = "sklearn.multiclass.fit_ovo is deprecated"
        raise NotImplementedError(msg)

    def predict_ovo(self, *args, **kwargs):
        """Deprecated"""
        msg = "sklearn.multiclass.predict_ovo is deprecated"
        raise NotImplementedError(msg)

    def fit_ecoc(self, *args, **kwargs):
        """Deprecated"""
        msg = "sklearn.multiclass.fit_ecoc is deprecated"
        raise NotImplementedError(msg)

    def predict_ecoc(self, *args, **kwargs):
        """Deprecated"""
        msg = "sklearn.multiclass.predict_ecoc is deprecated"
        raise NotImplementedError(msg)
