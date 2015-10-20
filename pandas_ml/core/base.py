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
    class BaseEstimator(object):
        pass

    class ClassifierMixin(object):
        pass

    class ClusterMixin(object):
        pass

    class RegressorMixin(object):
        pass

    class TransformerMixin(object):
        pass
