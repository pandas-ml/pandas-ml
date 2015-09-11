#!/usr/bin/env python

try:

    import sklearn.base as base

    BaseEstimator = base.BaseEstimator
    ClassifierMixin = base.ClassifierMixin
    ClusterMixin = base.ClusterMixin
    RegressorMixin = base.RegressorMixin
    TransformerMixin = base.TransformerMixin

except ImportError:

    # define dummy
    BaseEstimator = object
    ClassifierMixin = object
    ClusterMixin = object
    RegressorMixin = object
    TransformerMixin = object
