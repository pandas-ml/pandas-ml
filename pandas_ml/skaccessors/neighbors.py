#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods, _attach_methods, _wrap_data_func


class NeighborsMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.neighbors``.
    """

    _module_name = 'sklearn.neighbors'


_neighbor_methods = ['kneighbors_graph', 'radius_neighbors_graph']
_attach_methods(NeighborsMethods, _wrap_data_func, _neighbor_methods)
