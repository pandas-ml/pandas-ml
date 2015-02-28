#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class GridSearchMethods(AccessorMethods):
    _module_name = 'sklearn.grid_search'

    def describe(self, estimator):
        """
        Return cross validation results as ``pd.DataFrame``.
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
