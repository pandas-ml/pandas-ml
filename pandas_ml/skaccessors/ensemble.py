#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


class EnsembleMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.ensemble``.
    """

    _module_name = 'sklearn.ensemble'

    @property
    def partial_dependence(self):
        """Property to access ``sklearn.ensemble.partial_dependence``"""
        # unable to set with cache_readonly
        return PartialDependenceMethods(self._df)


class PartialDependenceMethods(_AccessorMethods):
    # _module_name = 'sklearn.ensemble.partial_dependence'
    # 'sklearn.ensemble.partial_dependence' has no attribute '__all__'

    def partial_dependence(self, gbrt, target_variables, **kwargs):
        """
        Call ``sklearn.ensemble.partial_dependence`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        import sklearn.ensemble.partial_dependence as pdp
        func = pdp.partial_dependence
        pdp, axes = func(gbrt, target_variables, X=self._data, **kwargs)
        return pdp, axes

    def plot_partial_dependence(self, gbrt, features, **kwargs):
        """
        Call ``sklearn.ensemble.plot_partial_dependence`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        """
        import sklearn.ensemble.partial_dependence as pdp
        func = pdp.plot_partial_dependence
        data = self._data
        fig, axes = func(gbrt, X=data, features=features, **kwargs)
        return fig, axes
