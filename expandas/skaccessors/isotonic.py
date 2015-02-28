#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.core.accessor import AccessorMethods, _attach_methods


class IsotonicMethods(AccessorMethods):
    _module_name = 'sklearn.isotonic'

    @property
    def IsotonicRegression(self):
        return self._module.IsotonicRegression

    def isotonic_regression(self, *args, **kwargs):
        func = self._module.isotonic_regression
        target = self.target
        _y = func(target.values, *args, **kwargs)
        _y = self._constructor_sliced(_y, index=target.index)
        return _y

    def check_increasing(self, *args, **kwargs):
        func = self._module.check_increasing
        target = self.target
        return func(target.index, target.values, *args, **kwargs)


