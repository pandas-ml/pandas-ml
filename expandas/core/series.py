#!/usr/bin/env python

import pandas as pd
from pandas.util.decorators import Appender, cache_readonly

import expandas.skaccessors as skaccessors


_shared_docs = dict()


class ModelSeries(pd.Series):
    """
    Wrapper for ``pandas.Series`` to support ``sklearn.preprocessing``
    """

    @property
    def _constructor(self):
        return ModelSeries

    # copied from expandas.core.frame
    _shared_docs['skaccessor'] = """
        Property to access ``sklearn.%(module)s``. See :mod:`expandas.skaccessors.%(module)s`
        """

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='preprocessing'))
    def preprocessing(self):
        return self._preprocessing

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='preprocessing'))
    def pp(self):
        return self._preprocessing

    @cache_readonly
    def _preprocessing(self):
        return skaccessors.PreprocessingMethods(self)

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
