#!/usr/bin/env python

import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.skaccessors import PreprocessingMethods


class ModelGeneric(object):
    """
    Define common accessor for ModelSeries and ModelFrame
    """

    def __init__(self):
        raise ValueError

    @cache_readonly
    def preprocessing(self):
        return PreprocessingMethods(self)
