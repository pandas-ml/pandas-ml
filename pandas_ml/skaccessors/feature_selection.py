#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods, _attach_methods, _wrap_data_target_func


class FeatureSelectionMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.feature_selection``.
    """

    _module_name = 'sklearn.feature_selection'


_fs_methods = ['chi2', 'f_classif', 'f_regression']
_attach_methods(FeatureSelectionMethods, _wrap_data_target_func, _fs_methods)
