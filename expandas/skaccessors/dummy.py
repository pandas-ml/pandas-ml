#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.core.accessor import AccessorMethods, _attach_methods


class DummyMethods(AccessorMethods):
    # _module_name = 'sklearn.dummy'
    # 'sklearn.dummy' has no attribute '__all__'

    @cache_readonly
    def DummyClassifier(self):
        import sklearn.dummy as dummy
        return dummy.DummyClassifier

    @cache_readonly
    def DummyRegressor(self):
        import sklearn.dummy as dummy
        return dummy.DummyRegressor
