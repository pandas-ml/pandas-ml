#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.core.accessor import AccessorMethods, _attach_methods


class EnsembleMethods(AccessorMethods):
    """
    Accessor to ``sklearn.ensemble``.
    """

    _module_name = 'sklearn.ensemble'

    @property
    def partial_dependence(self):
        # unable to set with cache_readonly
        return PartialDependenceMethods(self._df)


class PartialDependenceMethods(AccessorMethods):
    # _module_name = 'sklearn.ensemble.partial_dependence'
    # 'sklearn.ensemble.partial_dependence' has no attribute '__all__'

    def partial_dependence(self, gbrt, target_variables, **kwargs):
        import sklearn.ensemble.partial_dependence as pdp
        func = pdp.partial_dependence
        data = self.data
        pdp, axes = func(gbrt, target_variables, X=self.data, **kwargs)
        return pdp, axes

    def plot_partial_dependence(self, gbrt, features, **kwargs):
        import sklearn.ensemble.partial_dependence as pdp
        func = pdp.plot_partial_dependence
        data = self.data
        fig, axes = func(gbrt, X=data, features=features, **kwargs)
        return fig, axes
