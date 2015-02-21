#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.accessor import AccessorMethods, _attach_methods


class ClusterMethods(AccessorMethods):
    _module_name = 'sklearn.cluster'

    def k_means(self, n_clusters, *args, **kwargs):
        func = self._module.k_means
        data = self.data
        centroid, label, inertia = func(data.values, n_clusters, *args, **kwargs)
        label = pd.Series(label, index=data.index)
        return centroid, label, inertia

    def affinity_propagation(self, *args, **kwargs):
        func = self._module.affinity_propagation
        data = self.data
        cluster_centers_indices, labels = func(data.values, *args, **kwargs)
        labels = pd.Series(labels, index=data.index)
        return cluster_centers_indices, labels

    def dbscan(self, *args, **kwargs):
        func = self._module.dbscan
        data = self.data
        core_samples, labels = func(data.values, *args, **kwargs)
        labels = pd.Series(labels, index=data.index)
        return core_samples, labels

    def mean_shift(self, *args, **kwargs):
        func = self._module.mean_shift
        data = self.data
        cluster_centers, labels = func(data.values, *args, **kwargs)
        labels = pd.Series(labels, index=data.index)
        return cluster_centers, labels

    def spectral_clustering(self, *args, **kwargs):
        # raise NotImplementedError
        func = self._module.spectral_clustering
        data = self.data
        labels = func(data.values, *args, **kwargs)
        labels = pd.Series(labels, index=data.index)
        return labels

    # Biclustering

    @property
    def biclusterling(self):
        raise NotImplementedError

_cluster_methods = ['estimate_bandwidth', 'ward_tree']


def _wrap_func(func):
    def f(self, *args, **kwargs):
        data = self.data
        result = func(data.values, *args, **kwargs)
        # result = pd.Series(result, index=data.index)
        return result
    return f


_attach_methods(ClusterMethods, _wrap_func, _cluster_methods)


