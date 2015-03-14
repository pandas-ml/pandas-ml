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

    @Appender(pd.core.generic.NDFrame.groupby.__doc__)
    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False):
        from expandas.core.groupby import groupby
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        axis = self._get_axis_number(axis)
        return groupby(self, by=by, axis=axis, level=level, as_index=as_index,
                       sort=sort, group_keys=group_keys, squeeze=squeeze)
