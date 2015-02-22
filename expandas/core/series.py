#!/usr/bin/env python

import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.core.generic import ModelGeneric


class ModelSeries(ModelGeneric, pd.Series):
    """
    Wrapper to support preprocessing
    """

    @property
    def _constructor(self):
        return ModelSeries

    def __init__(self, *args, **kwargs):
        pd.Series.__init__(self, *args, **kwargs)

    pass
