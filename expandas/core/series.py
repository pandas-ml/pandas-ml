#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.util.decorators import Appender, cache_readonly

from expandas.core.generic import AbstractModel, _shared_docs
import expandas.skaccessors as skaccessors


class ModelSeries(pd.Series, AbstractModel):
    """
    Wrapper for ``pandas.Series`` to support ``sklearn.preprocessing``
    """

    @property
    def _constructor(self):
        return ModelSeries

    def _call(self, estimator, method_name, *args, **kwargs):
        method = self._check_attr(estimator, method_name)
        data = self.values
        result = method(data, *args, **kwargs)
        return result

    def _wrap_transform(self, transformed):
        """
        Wrapper for transform methods
        """
        if len(transformed.shape) == 2:
            if transformed.shape[0] != 1:
                from expandas.core.frame import ModelFrame
                return ModelFrame(transformed, index=self.index)
            else:
                transformed = transformed.flatten()
        return self._constructor(transformed, index=self.index, name=self.name)

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
