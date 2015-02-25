#!/usr/bin/env python

import warnings

import numpy as np
import pandas as pd
import pandas.compat as compat
from pandas.util.decorators import cache_readonly

from expandas.core.series import ModelSeries
import expandas.skaccessors as skaccessors


class ModelFrame(pd.DataFrame):

    _internal_names = (pd.core.generic.NDFrame._internal_names +
                       ['_target_name', '_estimator', '_predicted'])
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return ModelFrame

    _constructor_sliced = ModelSeries

    _TARGET_NAME = '.target'

    def __init__(self, data, target=None,
                 *args, **kwargs):
        try:
            from sklearn.datasets.base import Bunch
            if isinstance(data, Bunch):
                if target is not None:
                    raise ValueError
                # this should be first
                target = data.target
                # instanciate here to add column name
                columns = getattr(data, 'feature_names', None)
                data = pd.DataFrame(data.data, columns=columns)
        except ImportError:
            pass

        if isinstance(data, ModelFrame):
            target_name = data.target_name
        elif isinstance(target, pd.Series):
            target_name = target.name
            if target_name is None:
                target_name = self._TARGET_NAME
                target = pd.Series(target, name=target_name)
        else:
            target_name = self._TARGET_NAME

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, *args, **kwargs)

        if isinstance(target, compat.string_types):
            if target in data.columns:
                target_name = target
                df = data
            else:
                msg = "Specified target '{0}' is not included in data"
                raise ValueError(msg.format(target))

        elif target is None:
            df = data
        else:
            if not isinstance(target, pd.Series):
                target = pd.Series(target, name=target_name, index=data.index)
            df = self._concat_target(data, target)

        self._target_name = target_name
        self._estimator = None

        pd.DataFrame.__init__(self, df, *args, **kwargs)

    def _concat_target(self, data, target):
        assert isinstance(target, pd.Series)

        if len(data) != len(target):
            raise ValueError('data and target must have same length')

        if not data.index.equals(target.index):
            raise ValueError('data and target must have equal index')
        return pd.concat([target, data], axis=1)

    @property
    def data_columns(self):
        return pd.Index([c for c in self.columns if c != self.target_name])

    @property
    def data(self):
        return self.loc[:, self.data_columns]

    @data.setter
    def data(self, value):
        if value is None:
            del self.data
            return

        if isinstance(value, ModelFrame):
            if value.has_target():
                msg = 'Cannot update with {0} which has target attribute'
                raise ValueError(msg.format(self.__class__.__name__))

        if not isinstance(value, pd.DataFrame):
            value = pd.DataFrame(value, index=self.index)

        if self.target_name in value.columns:
            msg = "Passed data has the same column name as the target '{0}'"
            raise ValueError(msg.format(self.target_name))

        if self.has_target():
            value = self._concat_target(value, self.target)
        self._update_inplace(value)

    # don't allow to delete data

    def has_target(self):
        return self.target_name in self.columns

    @property
    def target_name(self):
        return self._target_name

    @target_name.setter
    def target_name(self, value):
        self._target_name = value

    @property
    def target(self):
        if self.has_target():
            return self.loc[:, self.target_name]
        else:
            msg = "{0} doesn't have target '{1}'"
            raise ValueError(msg.format(self.__class__.__name__, self.target_name))

    @target.setter
    def target(self, target):
        if target is None:
            del self.target
            return

        if not self.has_target():
            # allow to update target_name only when target attibute doesn't exist
            if isinstance(target, pd.Series):
                if target.name is not None:
                    self.target_name = target.name

        if isinstance(target, compat.string_types):
            if target in self.columns:
                self.target_name = target
            else:
                msg = "Specified target '{0}' is not included in data"
                raise ValueError(msg.format(target))
            return

        if isinstance(target, pd.Series):
            if target.name != self.target_name:
                msg = "Passed data is being renamed to '{0}'".format(self.target_name)
                warnings.warn(msg)
                target = pd.Series(target, name=self.target_name)
        else:
            target = pd.Series(target, index=self.index, name=self.target_name)

        df = self._concat_target(self.data, target)
        self._update_inplace(df)

    @target.deleter
    def target(self):
        self._update_inplace(self.data)

    @property
    def estimator(self):
        if self._estimator is None:
            raise ValueError('No estimator has been applied.')
        else:
            return self._estimator

    @property
    def predicted(self):
        return self._predicted

    def _check_attr(self, estimator, method_name):
        if not hasattr(estimator, method_name):
            msg = "class {0} doesn't have {1} method"
            raise ValueError(msg.format(type(estimator), method_name))
        return getattr(estimator, method_name)

    def _call(self, estimator, method_name, *args, **kwargs):
        method = self._check_attr(estimator, method_name)

        data = self.data.values
        if self.has_target():
            target = self.target.values
            try:
                result = method(data, y=target, *args, **kwargs)
            except TypeError:
                result = method(data, *args, **kwargs)
        else:
            # not try to pass target if it doesn't exists
            # to catch ValueError from estimator
            result = method(data, *args, **kwargs)
        return result

    def fit(self, estimator, *args, **kwargs):
        return self._call(estimator, 'fit', *args, **kwargs)

    def predict(self, estimator, *args, **kwargs):
        predicted = self._call(estimator, 'predict', *args, **kwargs)
        self._predicted = self._constructor_sliced(predicted, index=self.index)
        self._estimator = estimator
        return self._predicted

    def fit_predict(self, estimator, *args, **kwargs):
        predicted = self._call(estimator, 'fit_predict', *args, **kwargs)
        self._predicted = self._constructor_sliced(predicted, index=self.index)
        return self._predicted

    def score(self, estimator, *args, **kwargs):
        score = self._call(estimator, 'score', *args, **kwargs)
        return score

    def transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'transform', *args, **kwargs)

        if self.has_target():
            return self._constructor(transformed, target=self.target, index=self.index)
        else:
            return self._constructor(transformed, index=self.index)

    def fit_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'fit_transform', *args, **kwargs)

        if self.has_target():
            return self._constructor(transformed, target=self.target, index=self.index)
        else:
            return self._constructor(transformed, index=self.index)

    def inverse_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'inverse_transform', *args, **kwargs)

        if self.has_target():
            return self._constructor(transformed, target=self.target, index=self.index)
        else:
            return self._constructor(transformed, index=self.index)

    @cache_readonly
    def cluster(self):
        return skaccessors.ClusterMethods(self)

    @cache_readonly
    def covariance(self):
        return skaccessors.CovarianceMethods(self)

    @cache_readonly
    def cross_validation(self):
        return skaccessors.CrossValidationMethods(self)

    @cache_readonly
    def decomposition(self):
        return skaccessors.DecompositionMethods(self)

    @cache_readonly
    def dummy(self):
        return skaccessors.DummyMethods(self)

    @cache_readonly
    def ensemble(self):
        return skaccessors.EnsembleMethods(self)

    @cache_readonly
    def grid_search(self):
        return skaccessors.GridSearchMethods(self)

    @cache_readonly
    def lda(self):
        return skaccessors.LDAMethods(self)

    @cache_readonly
    def linear_model(self):
        return skaccessors.LinearModelMethods(self)

    @cache_readonly
    def feature_selection(self):
        return skaccessors.FeatureSelectionMethods(self)

    @cache_readonly
    def metrics(self):
        return skaccessors.MetricsMethods(self)

    @cache_readonly
    def naive_bayes(self):
        return skaccessors.NaiveBayesMethods(self)

    @cache_readonly
    def neighbors(self):
        return skaccessors.NeighborsMethods(self)

    @cache_readonly
    def pipeline(self):
        return skaccessors.PipelineMethods(self)

    @cache_readonly
    def preprocessing(self):
        return skaccessors.PreprocessingMethods(self)

    @cache_readonly
    def svm(self):
        return skaccessors.SVMMethods(self)

    @cache_readonly
    def tree(self):
        return skaccessors.TreeMethods(self)

