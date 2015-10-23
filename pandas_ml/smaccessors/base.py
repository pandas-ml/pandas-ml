#!/usr/bin/env python

import pandas as pd

from pandas_ml.core import base


def _maybe_statsmodels_data(data, target):
    try:
        import statsmodels.datasets as datasets
        if isinstance(data, datasets.utils.Dataset):
            if target is not None:
                raise ValueError("'target' can't be specified for sklearn.datasets")

            # this should be first
            try:
                target_name = getattr(data, 'endog_name', None)
                target = pd.Series(data.endog, name=target_name)
            except AttributeError:
                target = None
            try:
                columns = getattr(data, 'exog_name', None)
                if hasattr(data, 'censors'):
                    data = {columns[0]: data.censors, columns[1]: data.exog}
                    # hack for "heart" dataset
                    data = pd.DataFrame(data, columns=columns)
                else:
                    data = pd.DataFrame(data.exog, columns=columns)
            except AttributeError:
                raise ValueError("Unable to read statsmodels Dataset without exog")

            return data, target
    except ImportError:
        pass

    return data, target


class StatsModelsRegressor(base._BaseEstimator, base._RegressorMixin):

    def __init__(self, statsmodel=None, **parameters):
        self.statsmodel = statsmodel
        self.parameters = parameters

    def fit(self, X, y, *args, **kwargs):
        if self.statsmodel is None:
            import statsmodels.api as sm
            self.statsmodel = sm.OLS

        statsmodel = self.statsmodel(y, X, **self.parameters)
        self.fitted_ = statsmodel.fit(*args, **kwargs)
        return self.fitted_

    def predict(self, X, *args, **kwargs):
        try:
            return self.fitted_.predict(X, *args, **kwargs)
        except AttributeError:
            raise ValueError('{0} is not fitted to data'.format(self.__class__.__name__))

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        params = self.parameters.copy()
        params['statsmodel'] = self.statsmodel
        return params

    def set_params(self, **parameters):
        if 'statsmodel' in parameters:
            self.statsmodel = parameters.pop('statsmodel')
        try:
            # updated existing parameters
            self.parameters.update(parameters)
        except AttributeError:
            # parameters may not exists in case of grid search
            self.parameters = parameters
        # must return itself to make grid_search work
        return self
