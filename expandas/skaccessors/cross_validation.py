#!/usr/bin/env python

import warnings

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods


class CrossValidationMethods(AccessorMethods):
    """
    Accessor to ``sklearn.cross_validation``.
    """

    _module_name = 'sklearn.cross_validation'

    def StratifiedShuffleSplit(self, *args, **kwargs):
        """
        Instanciate ``sklearn.cross_validation.StratifiedShuffleSplit`` using automatic mapping.

        - ``y``: ``ModelFrame.target``
        """
        target = self._target
        return self._module.StratifiedShuffleSplit(target.values, *args, **kwargs)

    def iterate(self, cv):
        """
        Generate ``ModelFrame`` using iterators for cross validation

        Parameters
        ----------
        cv : cross validation iterator

        Returns
        -------
        generated : generator of ``ModelFrame``
        """
        if not(isinstance(cv, self._module._PartitionIterator)):
            msg = "{0} is not a subclass of PartitionIterator"
            warnings.warn(msg.format(cv.__class__.__name__))

        for train_index, test_index in cv:
            train_df = self._df.iloc[train_index, :]
            test_df = self._df.iloc[test_index, :]
            yield train_df, test_df

    def train_test_split(self, *args, **kwargs):
        """
        Call ``sklearn.cross_validation.train_test_split`` using automatic mapping.
        """
        func = self._module.train_test_split

        data = self._data
        if self._df.has_target():
            target = self._target
            tr_d, te_d, tr_l, te_l = func(data, target, *args, **kwargs)

            # Create DataFrame here to retain data and target names
            tr_d = pd.DataFrame(tr_d, columns=data.columns)
            te_d = pd.DataFrame(te_d, columns=data.columns)

            tr_l = pd.Series(tr_l, name=target.name)
            te_l = pd.Series(te_l, name=target.name)

            train_df = self._constructor(data=tr_d, target=tr_l)
            test_df = self._constructor(data=te_d, target=te_l)
            return train_df, test_df
        else:
            tr_d, te_d = func(data, *args, **kwargs)

            # Create DataFrame here to retain data and target names
            tr_d = pd.DataFrame(tr_d, columns=data.columns)
            te_d = pd.DataFrame(te_d, columns=data.columns)

            train_df = self._constructor(data=tr_d)
            train_df.target_name = self._df.target_name
            test_df = self._constructor(data=te_d)
            test_df.target_name = self._df.target_name
            return train_df, test_df

    def cross_val_score(self, estimator, *args, **kwargs):
        """
        Call ``sklearn.cross_validation.cross_val_score`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.cross_val_score
        return func(estimator, X=self._data, y=self._target, *args, **kwargs)

    def permutation_test_score(self, estimator, *args, **kwargs):
        """
        Call ``sklearn.cross_validation.permutation_test_score`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.permutation_test_score
        score, pscores, pvalue = func(estimator, X=self._data, y=self._target,
                                      *args, **kwargs)
        return score, pscores, pvalue

    def check_cv(self, cv, *args, **kwargs):
        """
        Call ``sklearn.cross_validation.check_cv`` using automatic mapping.

        - ``X``: ``ModelFrame.data``
        - ``y``: ``ModelFrame.target``
        """
        func = self._module.check_cv
        return func(cv, X=self._data, y=self._target, *args, **kwargs)

