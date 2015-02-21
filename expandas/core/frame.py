#!/usr/bin/env python

import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.skaccessors import (ClusterMethods,
                                  CrossValidationMethods,
                                  MetricsMethods,
                                  PreprocessingMethods)

class ModelFrame(pd.DataFrame):

    _internal_names = (pd.core.generic.NDFrame._internal_names +
                       ['target_name', 'estimators'])
    _internal_names_set = set(_internal_names)

    def __init__(self, data, target=None, target_name=None,
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
        except ImportError, e:
            print(e)
            pass

        if target_name is None:
            if isinstance(data, ModelFrame):
                target_name = data.target_name
            else:
                # default
                target_name = '.target'

        if target is None:
            df = data
        else:
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            if not isinstance(target, pd.Series):
                target = pd.Series(target, name=target_name)

            df = pd.concat([target, data], axis=1)

        self.target_name = target_name
        # estimator histories
        self.estimators = []

        super(ModelFrame, self).__init__(df, *args, **kwargs)

    @property
    def _constructor(self):
        return ModelFrame

    @property
    def data_columns(self):
        return pd.Index([c for c in self.columns if c != self.target_name])

    @property
    def data(self):
        return self.loc[:, self.data_columns]

    @property
    def target(self):
        return self.loc[:, self.target_name]

    @property
    def estimator(self):
        if len(self.estimators) > 0:
            # most recently used
            return self.estimators[-1]
        else:
            raise ValueError

    def _check_attr(self, estimator, func_name):
        if not hasattr(estimator, func_name):
            msg = "class {0} doesn't have {1} method"
            raise ValueError(msg.format(type(estimator), func_name))

    def fit(self, estimator, history=True, *args, **kwargs):
        self._check_attr(estimator, 'fit')

        try:
            estimator.fit(self.data.values, y=self.target, *args, **kwargs)
        except TypeError:
            estimator.fit(self.data.values, *args, **kwargs)

        if history:
            self.estimators.append(estimator)
        else:
            # only store latest
            self.estimators = [estimator]


    def predict(self, estimator, *args, **kwargs):
        self._check_attr(estimator, 'predict')
        predicted = estimator.predict(self.data, *args, **kwargs)

        self.predicted = pd.Series(predicted, index=self.index)
        return self.predicted

    def fit_predict(self, estimator, *args, **kwargs):
        self._check_attr(estimator, 'fit_predict')
        predicted = estimator.fit_predict(self.data, *args, **kwargs)

        self.predicted = pd.Series(predicted, index=self.index)
        return self.predicted

    def score(self, estimator, *args, **kwargs):
        self._check_attr(estimator, 'score')
        predicted = estimator.score(self.data, y=self.target, *args, **kwargs)

        self.predicted = pd.Series(predicted, index=self.index)
        return self.predicted

    def transform(self, estimator, *args, **kwargs):
        self._check_attr(estimator, 'transform')
        transformed = estimator.transform(self.data, y=self.target, *args, **kwargs)

        return pd.DataFrame(transformed, index=self.index)

    def fit_transform(self, estimator, *args, **kwargs):
        self._check_attr(estimator, 'fit_transform')
        transformed = estimator.fit_transform(self.data, y=self.target, *args, **kwargs)

        return pd.DataFrame(transformed, index=self.index)

    @cache_readonly
    def cluster(self):
        return ClusterMethods(self)

    @cache_readonly
    def cross_validation(self):
        return CrossValidationMethods(self)

    @property
    def decomposition(self):
        raise NotImplementedError

    @property
    def feature_selection(self):
        raise NotImplementedError

    @cache_readonly
    def metrics(self):
        return MetricsMethods(self)

    @cache_readonly
    def preprocessing(self):
        return PreprocessingMethods(self)
