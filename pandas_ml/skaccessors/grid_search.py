#!/usr/bin/env python

import numpy as np
import pandas as pd

from pandas_ml.core.accessor import AccessorMethods


class GridSearchMethods(AccessorMethods):
    """
    Accessor to ``sklearn.grid_search``.
    """

    _module_name = 'sklearn.grid_search'

    def describe(self, estimator):
        """
        Describe grid search results

        Parameters
        ----------
        estimator : fitted grid search estimator

        Returns
        -------
        described : ``ModelFrame``
        """
        results = []
        for params, mean_score, scores in estimator.grid_scores_:
            row = dict(mean=mean_score, std=scores.std())
            row.update(params)
            results.append(row)
        df = self._constructor(results)

        scores = pd.Index(['mean', 'std'])
        df = df[scores.append(df.columns[~df.columns.isin(scores)])]
        return df
