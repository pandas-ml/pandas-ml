#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.grid_search as gs

import expandas as expd
import expandas.util.testing as tm


class TestGridSearch(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.grid_search.GridSearchCV, gs.GridSearchCV)
        self.assertIs(df.grid_search.ParameterGrid, gs.ParameterGrid)
        self.assertIs(df.grid_search.ParameterSampler, gs.ParameterSampler)
        self.assertIs(df.grid_search.RandomizedSearchCV, gs.RandomizedSearchCV)

    def test_grid_search(self):
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100]},
                            {'kernel': ['linear'], 'C': [1, 10, 100]}]

        df = expd.ModelFrame(datasets.load_digits())
        cv = df.grid_search.GridSearchCV(df.svm.SVC(C=1), tuned_parameters, cv=5, scoring='precision')

        with tm.RNGContext(1):
            df.fit(cv)

        result = df.grid_search.describe(cv)
        expected = pd.DataFrame({'mean': [0.974108, 0.951416, 0.975372, 0.962534,  0.975372,
                                          0.964695, 0.951811, 0.951811, 0.951811],
                                 'std': [0.01313946, 0.02000999, 0.01128049, 0.0202183, 0.01128049,
                                         0.0166863, 0.01840967, 0.01840967, 0.01840967],
                                 'C': [1, 1, 10, 10, 100, 100, 1, 10, 100],
                                 'gamma': [0.001, 0.0001, 0.001, 0.0001, 0.001, 0.0001,
                                           np.nan, np.nan, np.nan],
                                 'kernel': ['rbf'] * 6 + ['linear'] * 3},
                                 columns=['mean', 'std', 'C', 'gamma', 'kernel'])
        self.assert_frame_equal(result, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
