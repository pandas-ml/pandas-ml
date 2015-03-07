#!/usr/bin/env python

import pandas as pd
from pandas.util.decorators import Appender, cache_readonly

import expandas.skaccessors as skaccessors


class ModelSeries(pd.Series):
    """
    Wrapper for ``pandas.Series`` to support ``sklearn.preprocessing``
    """

    @property
    def _constructor(self):
        return ModelSeries

    @cache_readonly
    def preprocessing(self):
        return skaccessors.PreprocessingMethods(self)

    @property
    def pp(self):
        """Property to access ``sklearn.preprocessing``"""
        return self.preprocessing

    @Appender(pd.Series.to_frame.__doc__)
    def to_frame(self, name=None):
        from expandas.core.frame import ModelFrame

        if name is None:
            name = self.name

        if name is None:
            df = ModelFrame(self)
        else:
            df = ModelFrame({name: self})

        return df
