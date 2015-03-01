#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class NeighborsMethods(AccessorMethods):
    """
    Accessor to ``sklearn.neighbors``.
    """

    _module_name = 'sklearn.neighbors'


_lm_methods = ['kneighbors_graph', 'radius_neighbors_graph']


def _wrap_func(func):
    def f(self, *args, **kwargs):
        data = self.data
        result = func(data.values, *args, **kwargs)
        return result
    return f


_attach_methods(NeighborsMethods, _wrap_func, _lm_methods)


