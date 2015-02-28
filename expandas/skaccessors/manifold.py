#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class ManifoldMethods(AccessorMethods):
    _module_name = 'sklearn.manifold'

    def locally_linear_embedding(self, n_neighbors, n_components, *args, **kwargs):
        func = self._module.locally_linear_embedding
        y, squared_error = func(self.data.values, n_neighbors,
                                n_components,  *args, **kwargs)
        y = self._constructor(y, index=self._df.index)
        return y, squared_error

    def spectral_embedding(self, *args, **kwargs):
        func = self._module.spectral_embedding
        data = self.data
        embedding = func(data.values, *args, **kwargs)
        embedding = self._constructor(embedding, index=data.index)
        return embedding
