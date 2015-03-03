#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class MetricsMethods(AccessorMethods):
    """
    Accessor to ``sklearn.metrics``.
    """

    _module_name = 'sklearn.metrics'

    # Model Selection Interface
    # ``make_scorer`` will be attached via autoload

    # Clasification metrics

    def auc(self, *args, **kwargs):
        raise NotImplementedError
        """
        # should work, but depends on roc_curve
        func = self._module.auc
        result = func(self.data.values, self.target.values,
                      *args, **kwargs)
        return result
        """

    def average_precision_score(self, *args, **kwargs):
        raise NotImplementedError

    # def test_classification_report
    # Consider whether to return DataFrame rather than text

    def confusion_matrix(self, *args, **kwargs):
        func = self._module.confusion_matrix
        result = func(self.target.values, self.predicted.values,
                      *args, **kwargs)
        result = self._constructor(result)
        result.index.name = 'Target'
        result.columns.name = 'Predicted'
        return result

    def hinge_loss(self, *args, **kwargs):
        raise NotImplementedError

    def log_loss(self, *args, **kwargs):
        raise NotImplementedError

    def precision_recall_curve(self, *args, **kwargs):
        raise NotImplementedError

    def precision_recall_fscore_support(self, *args, **kwargs):
        func = self._module.precision_recall_fscore_support
        precision, recall, fscore, support = func(self.target.values,
                                                  self.predicted.values,
                                                  *args, **kwargs)
        result = self._constructor({'precision': precision, 'recall': recall,
                                    'f1-score': fscore, 'support': support},
                                   columns=['precision', 'recall', 'f1-score', 'support'])
        return result

    def roc_auc_score(self, *args, **kwargs):
        raise NotImplementedError

    def roc_curve(self, *args, **kwargs):
        # should return DataFrame
        raise NotImplementedError

    # Regression metrics
    # None

    # Clusteing metrics

    def silhouette_score(self, *args, **kwargs):
        func = self._module.silhouette_score
        result = func(self.data.values, self.predicted.values,
                      *args, **kwargs)
        return result

    def silhouette_samples(self, *args, **kwargs):
        func = self._module.silhouette_samples
        result = func(self.data.values, self.predicted.values,
                      *args, **kwargs)
        result = self._constructor_sliced(result, index=self._df.index)
        return result

    # Biclustering metrics

    def consensus_score(self, *args, **kwargs):
        raise NotImplementedError

    # Pairwise metrics

    @property
    def pairwise(self):
        raise NotImplementedError


# y_true and y_pred
_classification_methods = ['accuracy_score', 'classification_report',
                           'f1_score', 'fbeta_score',
                           'hamming_loss', 'jaccard_similarity_score',
                           'matthews_corrcoef', 'precision_score',
                           'recall_score', 'zero_one_loss']

_regression_methods = ['explained_variance_score', 'mean_absolute_error',
                       'mean_squared_error', 'r2_score']

_cluster_methods = ['adjusted_mutual_info_score',
                    'adjusted_rand_score',
                    'completeness_score',
                    'homogeneity_completeness_v_measure',
                    'homogeneity_score',
                    'mutual_info_score',
                    'normalized_mutual_info_score',
                    'v_measure_score']

_true_pred_methods = (_classification_methods + _regression_methods +
                      _cluster_methods)


def _wrap_func(func):
    def f(self, *args, **kwargs):
        result = func(self.target.values,
                      self.predicted.values,
                      *args, **kwargs)
        return result
    return f


_attach_methods(MetricsMethods, _wrap_func, _true_pred_methods)

