#!/usr/bin/env python

import warnings

import numpy as np
import pandas as pd
import pandas.core.common as com
import pandas.compat as compat
from pandas.util.decorators import Appender, cache_readonly

from expandas.core.series import ModelSeries
from expandas.core.accessor import AccessorMethods
import expandas.skaccessors as skaccessors
import expandas.misc as misc


_shared_docs = dict()


class ModelFrame(pd.DataFrame):
    """
    Data structure subclassing ``pandas.DataFrame`` to define a metadata to
    specify target (response variable) and data (explanatory variable / features).

    Parameters
    ----------
    data : same as ``pandas.DataFrame``
    target : str or array-like
        Column name or values to be used as target
    args : arguments passed to ``pandas.DataFrame``
    kwargs : keyword arguments passed to ``pandas.DataFrame``
    """

    _internal_names = (pd.core.generic.NDFrame._internal_names +
                       ['_target_name', '_estimator',
                        '_predicted', '_proba', '_log_proba', '_decision'])
    _internal_names_set = set(_internal_names)

    _mapper = dict(fit=dict(),
                   predict={'GaussianProcess': skaccessors.GaussianProcessMethods._predict})

    @property
    def _constructor(self):
        return ModelFrame

    _constructor_sliced = ModelSeries

    _TARGET_NAME = '.target'

    def __init__(self, data, target=None,
                 *args, **kwargs):

        if data is None and target is None:
            msg = '{0} must have either data or target'
            raise ValueError(msg.format(self.__class__.__name__))
        elif data is None and not com.is_list_like(target):
            msg = 'target must be list-like when data is None'
            raise ValueError(msg)

        data, target = skaccessors._maybe_sklearn_data(data, target)

        # retrieve target_name
        if isinstance(data, ModelFrame):
            target_name = data.target_name
        elif isinstance(target, pd.Series):
            target_name = target.name
            if target_name is None:
                target_name = self._TARGET_NAME
                target = pd.Series(target, name=target_name)
        else:
            target_name = self._TARGET_NAME

        data, target = self._maybe_convert_data(data, target, target_name, *args, **kwargs)

        if target is not None and not com.is_list_like(target):
            if target in data.columns:
                target_name = target
                df = data
            else:
                msg = "Specified target '{0}' is not included in data"
                raise ValueError(msg.format(target))
        else:
            df = self._concat_target(data, target)

        self._target_name = target_name
        self._estimator = None
        self._predicted = None
        self._proba = None
        self._log_proba = None
        self._decision = None

        pd.DataFrame.__init__(self, df, *args, **kwargs)

    def _maybe_convert_data(self, data, target, target_name, *args, **kwargs):
        """
        Internal function to instanciate data and target

        Parameters
        ----------
        data : instance converted to ``pandas.DataFrame``
        target : instance converted to ``pandas.Series``
        args : argument passed from ``__init__``
        kwargs : argument passed from ``__init__``
        """

        init_df = isinstance(data, pd.DataFrame)
        init_target = isinstance(target, pd.Series)

        if not init_df and not init_target:
            if data is not None:
                data = pd.DataFrame(data, *args, **kwargs)

            if com.is_list_like(target):
                if data is not None:
                    target = pd.Series(target, name=target_name, index=data.index)
                else:
                    target = pd.Series(target, name=target_name)
        elif not init_df:
            if data is not None:
                index = kwargs.pop('index', target.index)
                data = pd.DataFrame(data, index=index, *args, **kwargs)
        elif not init_target:
            if com.is_list_like(target):
                if data is not None:
                    target = pd.Series(target, name=target_name, index=data.index)
                else:
                    target = pd.Series(target, name=target_name)
        else:
            # no conversion required
            pass
        return data, target


    def _concat_target(self, data, target):
        if data is None and target is None:
            msg = '{0} must have either data or target'
            raise ValueError(msg.format(self.__class__.__name__))

        elif data is None:
            return target

        elif target is None:
            return data

        if len(data) != len(target):
            raise ValueError('data and target must have same length')

        if not data.index.equals(target.index):
            raise ValueError('data and target must have equal index')
        return pd.concat([target, data], axis=1)

    def has_data(self):
        """
        Return whether ``ModelFrame`` has data

        Returns
        -------
        has_data : bool
        """
        return len(self._data_columns) > 0

    @property
    def _data_columns(self):
        return pd.Index([c for c in self.columns if c != self.target_name])

    @property
    def data(self):
        """
        Return data (explanatory variable / features)

        Returns
        -------
        data : ``ModelFrame``
        """
        if self.has_data():
            return self.loc[:, self._data_columns]
        else:
            return None

    @data.setter
    def data(self, data):
        if data is None:
            del self.data
            return

        if isinstance(data, ModelFrame):
            if data.has_target():
                msg = 'Cannot update with {0} which has target attribute'
                raise ValueError(msg.format(self.__class__.__name__))

        data, _ = self._maybe_convert_data(data, self.target, self.target_name)

        if self.target_name in data.columns:
            msg = "Passed data has the same column name as the target '{0}'"
            raise ValueError(msg.format(self.target_name))

        if self.has_target():
            data = self._concat_target(data, self.target)
        self._update_inplace(data)

    @data.deleter
    def data(self):
        if self.has_target():
            self._update_inplace(self.target.to_frame())
        else:
            msg = '{0} must have either data or target'
            raise ValueError(msg.format(self.__class__.__name__))

    def has_target(self):
        """
        Return whether ``ModelFrame`` has target

        Returns
        -------
        has_target : bool
        """
        return self.target_name in self.columns

    @property
    def target_name(self):
        """
        Return target column name

        Returns
        -------
        target : object
        """
        return self._target_name

    @target_name.setter
    def target_name(self, value):
        self._target_name = value

    @property
    def target(self):
        """
        Return target (response variable)

        Returns
        -------
        target : ``ModelSeries``
        """
        if self.has_target():
            return self.loc[:, self.target_name]
        else:
            return None

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

        if not com.is_list_like(target):
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

            _, target = self._maybe_convert_data(self.data, target, self.target_name)

        df = self._concat_target(self.data, target)
        self._update_inplace(df)

    @target.deleter
    def target(self):
        if self.has_data():
            self._update_inplace(self.data)
        else:
            msg = '{0} must have either data or target'
            raise ValueError(msg.format(self.__class__.__name__))

    @property
    def estimator(self):
        """
        Return most recently used estimator

        Returns
        -------
        estimator : estimator
        """
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        if not self._estimator is value:
            self._estimator = value
            # reset other properties
            self._predicted = None
            self._proba = None
            self._log_proba = None
            self._decision = None

    @property
    def predicted(self):
        """
        Return current estimator's predicted results

        Returns
        -------
        predicted : ``ModelSeries``
        """
        if self._predicted is None:
            self._predicted = self.predict(self.estimator)
            msg = "Automatically call '{0}.predict()'' to get predicted results"
            warnings.warn(msg.format(self.estimator.__class__.__name__))
        return self._predicted

    @property
    def proba(self):
        """
        Return current estimator's probabilities

        Returns
        -------
        probabilities : ``ModelFrame``
        """
        if self._proba is None:
            self._proba = self.predict_proba(self.estimator)
            msg = "Automatically call '{0}.predict_proba()' to get probabilities"
            warnings.warn(msg.format(self.estimator.__class__.__name__))
        return self._proba

    @property
    def log_proba(self):
        """
        Return current estimator's log probabilities

        Returns
        -------
        probabilities : ``ModelFrame``
        """
        if self._log_proba is None:
            self._log_proba = self.predict_log_proba(self.estimator)
            msg = "Automatically call '{0}.predict_log_proba()' to get log probabilities"
            warnings.warn(msg.format(self.estimator.__class__.__name__))
        return self._log_proba

    @property
    def decision(self):
        """
        Return current estimator's decision function

        Returns
        -------
        decisions : ``ModelFrame``
        """
        if self._decision is None:
            self._decision = self.decision_function(self.estimator)
            msg = "Automatically call '{0}.decition_function()' to get decision function"
            warnings.warn(msg.format(self.estimator.__class__.__name__))
        return self._decision

    def _check_attr(self, estimator, method_name):
        if not hasattr(estimator, method_name):
            msg = "class {0} doesn't have {1} method"
            raise ValueError(msg.format(type(estimator), method_name))
        return getattr(estimator, method_name)

    def _get_mapper(self, estimator, method_name):
        if method_name in self._mapper:
            mapper = self._mapper[method_name]
        return mapper.get(estimator.__class__.__name__, None)

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
        self.estimator = estimator
        return result

    _shared_docs['estimator_methods'] = """
        Call estimator's %(funcname)s method.

        Parameters
        ----------
        args : arguments passed to %(funcname)s method
        kwargs : keyword arguments passed to %(funcname)s method

        Returns
        -------
        %(returned)s
        """

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit', returned='returned : None or fitted estimator'))
    def fit(self, estimator, *args, **kwargs):
        return self._call(estimator, 'fit', *args, **kwargs)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='predict', returned='returned : predicted result'))
    def predict(self, estimator, *args, **kwargs):
        mapped = self._get_mapper(estimator, 'predict')
        if mapped is not None:
            result = mapped(self, estimator, *args, **kwargs)
            # save estimator when succeeded
            self.estimator = estimator
            return result
        predicted = self._call(estimator, 'predict', *args, **kwargs)
        return self._wrap_predicted(predicted, estimator)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit_predict', returned='returned : predicted result'))
    def fit_predict(self, estimator, *args, **kwargs):
        predicted = self._call(estimator, 'fit_predict', *args, **kwargs)
        return self._wrap_predicted(predicted, estimator)

    def _wrap_predicted(self, predicted, estimator):
        """
        Wrapper for predict methods
        """
        try:
            predicted = self._constructor_sliced(predicted, index=self.index)
        except ValueError:
            msg = "Unable to instantiate ModelSeries for '{0}'"
            warnings.warn(msg.format(estimator.__class__.__name__))
        self._predicted = predicted
        return self._predicted

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='predict_proba', returned='returned : probabilities'))
    def predict_proba(self, estimator, *args, **kwargs):
        probability = self._call(estimator, 'predict_proba', *args, **kwargs)
        self._proba = self._wrap_probability(probability, estimator)
        return self._proba

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='predict_log_proba', returned='returned : probabilities'))
    def predict_log_proba(self, estimator, *args, **kwargs):
        probability = self._call(estimator, 'predict_log_proba', *args, **kwargs)
        self._log_proba = self._wrap_probability(probability, estimator)
        return self._log_proba

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='decision_function', returned='returned : decisions'))
    def decision_function(self, estimator, *args, **kwargs):
        decision = self._call(estimator, 'decision_function', *args, **kwargs)
        self._decision = self._wrap_probability(decision, estimator)
        return self._decision

    def _wrap_probability(self, probability, estimator):
        """
        Wrapper for probability methods
        """
        try:
            if len(probability.shape) < 2 or probability.shape[1] == 1:
                # 2 class
                probability = self._constructor(probability, index=self.index)
            else:
                probability = self._constructor(probability, index=self.index,
                                                columns=estimator.classes_)
        except ValueError:
            msg = "Unable to instantiate ModelFrame for '{0}'"
            warnings.warn(msg.format(estimator.__class__.__name__))
        return probability

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='score', returned='returned : score'))
    def score(self, estimator, *args, **kwargs):
        score = self._call(estimator, 'score', *args, **kwargs)
        return score

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='transform', returned='returned : transformed result'))
    def transform(self, estimator, *args, **kwargs):
        if isinstance(estimator, compat.string_types):
            return misc.transform_with_patsy(estimator, self, *args, **kwargs)
        transformed = self._call(estimator, 'transform', *args, **kwargs)
        return self._wrap_transform(transformed)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit_transform', returned='returned : transformed result'))
    def fit_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'fit_transform', *args, **kwargs)
        return self._wrap_transform(transformed)

    def _wrap_transform(self, transformed):
        """
        Wrapper for transform methods
        """
        if self.has_target():
            return self._constructor(transformed, target=self.target, index=self.index)
        else:
            return self._constructor(transformed, index=self.index)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='inverse_transform', returned='returned : transformed result'))
    def inverse_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'inverse_transform', *args, **kwargs)
        return self._wrap_transform(transformed)

    _shared_docs['skaccessor'] = """
        Property to access ``sklearn.%(module)s``. See :mod:`expandas.skaccessors.%(module)s`
        """

    _shared_docs['skaccessor_nolink'] = """
        Property to access ``sklearn.%(module)s``
        """

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='cluster'))
    def cluster(self):
        return self._cluster

    @cache_readonly
    def _cluster(self):
        return skaccessors.ClusterMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='covariance'))
    def covariance(self):
        return self._covariance

    @cache_readonly
    def _covariance(self):
        return skaccessors.CovarianceMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='cross_decomposition'))
    def cross_decomposition(self):
        return self._cross_decomposition

    @cache_readonly
    def _cross_decomposition(self):
        attrs = ['PLSRegression', 'PLSCanonical', 'CCA', 'PLSSVD']
        return AccessorMethods(self, module_name='sklearn.cross_decomposition',
                               attrs=attrs)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='cross_validation'))
    def cross_validation(self):
        return self._cross_validation

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='cross_validation'))
    def crv(self):
        return self._cross_validation

    @cache_readonly
    def _cross_validation(self):
        return skaccessors.CrossValidationMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='decomposition'))
    def decomposition(self):
        return self._decomposition

    @cache_readonly
    def _decomposition(self):
        return skaccessors.DecompositionMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='dummy'))
    def dummy(self):
        return self._dummy

    @cache_readonly
    def _dummy(self):
        attrs = ['DummyClassifier', 'DummyRegressor']
        return AccessorMethods(self, module_name='sklearn.dummy', attrs=attrs)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='ensemble'))
    def ensemble(self):
        return self._ensemble

    @cache_readonly
    def _ensemble(self):
        return skaccessors.EnsembleMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='feature_extraction'))
    def feature_extraction(self):
        return self._feature_extraction

    @cache_readonly
    def _feature_extraction(self):
        return skaccessors.FeatureExtractionMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='feature_selection'))
    def feature_selection(self):
        return self._feature_selection

    @cache_readonly
    def _feature_selection(self):
        return skaccessors.FeatureSelectionMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='gaussian_process'))
    def gaussian_process(self):
        return self._gaussian_process

    @cache_readonly
    def _gaussian_process(self):
        return skaccessors.GaussianProcessMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='grid_search'))
    def grid_search(self):
        return self._grid_search

    @cache_readonly
    def _grid_search(self):
        return skaccessors.GridSearchMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='isotonic'))
    def isotonic(self):
        return self._isotonic

    @cache_readonly
    def _isotonic(self):
        return skaccessors.IsotonicMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='kernel_approximation'))
    def kernel_approximation(self):
        return self._kernel_approximation

    @cache_readonly
    def _kernel_approximation(self):
        attrs = ['AdditiveChi2Sampler', 'Nystroem', 'RBFSampler', 'SkewedChi2Sampler']
        return AccessorMethods(self, module_name='sklearn.kernel_approximation',
                               attrs=attrs)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='lda'))
    def lda(self):
        return self._lda

    @cache_readonly
    def _lda(self):
        return AccessorMethods(self, module_name='sklearn.lda')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='learning_curve'))
    def learning_curve(self):
        return self._learning_curve

    @cache_readonly
    def _learning_curve(self):
        return skaccessors.LearningCurveMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='linear_model'))
    def linear_model(self):
        return self._linear_model

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='linear_model'))
    def lm(self):
        return self._linear_model

    @cache_readonly
    def _linear_model(self):
        return skaccessors.LinearModelMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='manifold'))
    def manifold(self):
        return self._manifold

    @cache_readonly
    def _manifold(self):
        return skaccessors.ManifoldMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='metrics'))
    def metrics(self):
        return self._metrics

    @cache_readonly
    def _metrics(self):
        return skaccessors.MetricsMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='mixture'))
    def mixture(self):
        return self._mixture

    @cache_readonly
    def _mixture(self):
        return AccessorMethods(self, module_name='sklearn.mixture')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='multiclass'))
    def multiclass(self):
        return self._multiclass

    @cache_readonly
    def _multiclass(self):
        return skaccessors.MultiClassMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='naive_bayes'))
    def naive_bayes(self):
        return self._naive_bayes

    @cache_readonly
    def _naive_bayes(self):
        return AccessorMethods(self, module_name='sklearn.naive_bayes')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='neighbors'))
    def neighbors(self):
        return self._neighbors

    @cache_readonly
    def _neighbors(self):
        return skaccessors.NeighborsMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='neural_network'))
    def neural_network(self):
        return self._neural_network

    @cache_readonly
    def _neural_network(self):
        return AccessorMethods(self, module_name='sklearn.neural_network')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='pipeline'))
    def pipeline(self):
        return self._pipeline

    @cache_readonly
    def _pipeline(self):
        return skaccessors.PipelineMethods(self)

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='preprocessing'))
    def preprocessing(self):
        return self._preprocessing

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='preprocessing'))
    def pp(self):
        return self.preprocessing

    @cache_readonly
    def _preprocessing(self):
        return skaccessors.PreprocessingMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='qda'))
    def qda(self):
        return self._qda

    @cache_readonly
    def _qda(self):
        return AccessorMethods(self, module_name='sklearn.qda')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='random_projection'))
    def random_projection(self):
        return self._random_projection

    @cache_readonly
    def _random_projection(self):
        return AccessorMethods(self, module_name='sklearn.random_projection')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='semi_supervised'))
    def semi_supervised(self):
        return self._semi_supervised

    @cache_readonly
    def _semi_supervised(self):
        return AccessorMethods(self, module_name='sklearn.semi_supervised')

    @property
    @Appender(_shared_docs['skaccessor'] % dict(module='svm'))
    def svm(self):
        return self._svm

    @cache_readonly
    def _svm(self):
        return skaccessors.SVMMethods(self)

    @property
    @Appender(_shared_docs['skaccessor_nolink'] % dict(module='tree'))
    def tree(self):
        return self._tree

    @cache_readonly
    def _tree(self):
        return AccessorMethods(self, module_name='sklearn.tree')

    @Appender(pd.core.generic.NDFrame.groupby.__doc__)
    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False):
        from expandas.core.groupby import groupby
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        axis = self._get_axis_number(axis)
        return groupby(self, by=by, axis=axis, level=level, as_index=as_index,
                       sort=sort, group_keys=group_keys, squeeze=squeeze)
