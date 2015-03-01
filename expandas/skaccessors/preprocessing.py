#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class PreprocessingMethods(AccessorMethods):
    """
    Accessor to ``sklearn.preprocessing``.
    """

    _module_name = 'sklearn.preprocessing'

    def add_dummy_feature(self, value=1.0):
        from expandas.core.series import ModelSeries
        from expandas.core.frame import ModelFrame

        func = self._module.add_dummy_feature

        if isinstance(self._df, ModelSeries):
            data = self._df.to_frame()
            constructor = ModelFrame
        else:
            data = self.data
            constructor = self._constructor

        result = func(data.values, value=value)
        result = constructor(result, index=data.index)
        columns = result.columns[:-len(data.columns)].append(data.columns)
        result.columns = columns
        return result


_preprocessing_methods = ['binarize', 'normalize', 'scale']

def _wrap_func(func):
    def f(self, *args, **kwargs):
        from expandas.core.frame import ModelFrame
        if isinstance(self._df, ModelFrame):
            data = self.data
            result = func(data.values, *args, **kwargs)
            result = self._constructor(result, index=data.index,
                                       columns=data.columns)
        else:
            # ModelSeries
            values = np.atleast_2d(self._df.values)
            result = func(values, *args, **kwargs)
            result = self._constructor(result[0], index=self._df.index,
                                       name=self._df.name)
        return result
    return f


_attach_methods(PreprocessingMethods, _wrap_func, _preprocessing_methods)
