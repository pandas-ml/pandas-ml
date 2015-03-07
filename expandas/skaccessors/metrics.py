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

    def auc(self, kind='roc', reorder=False, **kwargs):
        func = self._module.auc
        if kind == 'roc':
            return self.roc_auc_score(**kwargs)
        elif kind == 'precision_recall_curve':
            return self.average_precision_score(**kwargs)
        else:
            msg = "Invalid kind: {0}, kind must be either 'roc' or 'precision_recall_curve'"
            raise ValueError(msf.format(kind))

    def average_precision_score(self, *args, **kwargs):
        func = self._module.average_precision_score
        return self._score_wraps(func, self._decision, *args, **kwargs)

    def confusion_matrix(self, *args, **kwargs):
        func = self._module.confusion_matrix
        result = func(self._target.values, self._predicted.values,
                      *args, **kwargs)
        result = self._constructor(result)
        result.index.name = 'Target'
        result.columns.name = 'Predicted'
        return result

    def f1_score(self, *args, **kwargs):
        func = self._module.f1_score
        return self._score_wraps(func, self._predicted, *args, **kwargs)

    def fbeta_score(self, beta, *args, **kwargs):
        func = self._module.fbeta_score
        return self._score_wraps(func, self._predicted, beta, *args, **kwargs)

    def _score_wraps(self, func, scorerer, *args, **kwargs):
        average = kwargs.get('average', 'weighted')
        result = func(self._target.values, scorerer.values, *args, **kwargs)
        if average is None:
            result = self._constructor_sliced(result)
        return result

    def hinge_loss(self, *args, **kwargs):
        func = self._module.hinge_loss
        result = func(self._target.values,
                      self._df.decision.values, *args, **kwargs)
        return result

    def log_loss(self, *args, **kwargs):
        func = self._module.log_loss
        result = func(self._target.values,
                      self._df.proba.values, *args, **kwargs)
        return result

    def precision_recall_curve(self, *args, **kwargs):
        func = self._module.precision_recall_curve
        return self._curve_wraps(func, *args, **kwargs)

    def _curve_wraps(self, func, *args, **kwargs):
        decision = self._df.decision
        if len(decision.shape) < 2 or decision.shape[1] == 1:
            c1, c2, threshold = func(self._target.values, decision.values,
                                     *args, **kwargs)
            return c1, c2, threshold

        results = {}
        for i, (name, col) in enumerate(decision.iteritems()):
            # results can have different length
            c1, c2, threshold = func(self._target.values, col.values,
                                     pos_label=i, *args, **kwargs)
            results[name] = c1, c2, threshold
        return results

    def precision_recall_fscore_support(self, *args, **kwargs):
        func = self._module.precision_recall_fscore_support
        p, r, f, s = func(self._target.values, self._predicted.values,
                          *args, **kwargs)
        result = self._constructor({'precision': p, 'recall': r,
                                    'f1-score': f, 'support': s},
                                   columns=['precision', 'recall', 'f1-score', 'support'])
        return result

    def precision_score(self, *args, **kwargs):
        func = self._module.precision_score
        return self._score_wraps(func, self._predicted, *args, **kwargs)

    def recall_score(self, *args, **kwargs):
        func = self._module.recall_score
        return self._score_wraps(func, self._predicted, *args, **kwargs)

    def roc_auc_score(self, *args, **kwargs):
        func = self._module.roc_auc_score
        return self._score_wraps(func, self._decision, *args, **kwargs)

    def roc_curve(self, *args, **kwargs):
        func = self._module.roc_curve
        return self._curve_wraps(func, *args, **kwargs)

    # Regression metrics
    # None

    # Clusteing metrics

    def silhouette_score(self, *args, **kwargs):
        func = self._module.silhouette_score
        result = func(self._data.values, self._predicted.values,
                      *args, **kwargs)
        return result

    def silhouette_samples(self, *args, **kwargs):
        func = self._module.silhouette_samples
        result = func(self._data.values, self._predicted.values,
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
                           'hamming_loss', 'jaccard_similarity_score',
                           'matthews_corrcoef', 'zero_one_loss']

_regression_methods = ['explained_variance_score', 'mean_absolute_error',
                       'mean_squared_error', 'r2_score']

_cluster_methods = ['mutual_info_score']

_true_pred_methods = (_classification_methods + _regression_methods +
                      _cluster_methods)


def _wrap_func(func):
    def f(self, *args, **kwargs):
        result = func(self._target.values, self._predicted.values,
                      *args, **kwargs)
        return result
    return f


_attach_methods(MetricsMethods, _wrap_func, _true_pred_methods)


# methods which doesn't take additional arguments
_cluster_methods_noargs = ['adjusted_mutual_info_score',
                           'adjusted_rand_score',
                           'completeness_score',
                           'homogeneity_completeness_v_measure',
                           'homogeneity_score',
                           'normalized_mutual_info_score',
                           'v_measure_score']


def _wrap_func_noargs(func):
    def f(self):
        result = func(self._target.values, self._predicted.values)
        return result
    return f


_attach_methods(MetricsMethods, _wrap_func_noargs, _cluster_methods_noargs)