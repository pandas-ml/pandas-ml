#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class FeatureSelectionMethods(AccessorMethods):
    _module_name = 'sklearn.feature_selection'


_lm_methods = ['chi2', 'f_classif', 'f_regression']


def _wrap_func(func):
    def f(self, *args, **kwargs):
        data = self.data
        target = self.target
        result = func(data.values, y=target.values, *args, **kwargs)
        return result
    return f


_attach_methods(FeatureSelectionMethods, _wrap_func, _lm_methods)


