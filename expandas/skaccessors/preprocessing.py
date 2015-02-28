#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class PreprocessingMethods(AccessorMethods):
    _module_name = 'sklearn.preprocessing'


_preprocessing_methods = ['add_dummy_feature',
                          'binarize', 'normalize', 'scale']

def _wrap_func(func):
    def f(self, *args, **kwargs):
        data = self.data
        result = func(data, *args, **kwargs)
        result = self._constructor(result, index=data.index,
                                   columns=data.columns)
        return result
    return f


_attach_methods(PreprocessingMethods, _wrap_func, _preprocessing_methods)
