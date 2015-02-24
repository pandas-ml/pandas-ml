#!/usr/bin/env python

import pandas as pd
from pandas.util.decorators import cache_readonly

import expandas.skaccessors as skaccessors


class ModelSeries(pd.Series):
    """
    Wrapper to support preprocessing
    """

    @property
    def _constructor(self):
        return ModelSeries

    @cache_readonly
    def preprocessing(self):
        return skaccessors.PreprocessingMethods(self)

