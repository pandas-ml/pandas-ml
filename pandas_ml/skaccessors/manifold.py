#!/usr/bin/env python


from pandas_ml.core.accessor import _AccessorMethods


class ManifoldMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.manifold``.
    """

    _module_name = 'sklearn.manifold'

    def locally_linear_embedding(self, n_neighbors, n_components, *args, **kwargs):
        """
        Call ``sklearn.manifold.locally_linear_embedding`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        func = self._module.locally_linear_embedding
        y, squared_error = func(self._data.values, n_neighbors,
                                n_components, *args, **kwargs)
        y = self._constructor(y, index=self._df.index)
        return y, squared_error

    def spectral_embedding(self, *args, **kwargs):
        """
        Call ``sklearn.manifold.spectral_embedding`` using automatic mapping.

        - ``adjacency``: ``ModelFrame.data``
        """
        func = self._module.spectral_embedding
        data = self._data
        embedding = func(data.values, *args, **kwargs)
        embedding = self._constructor(embedding, index=data.index)
        return embedding
