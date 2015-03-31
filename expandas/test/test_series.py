#!/usr/bin/env python

import datetime
import warnings

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.svm as svm
import sklearn.preprocessing as pp

import expandas as expd
import expandas.util.testing as tm


class TestModelSeries(tm.TestCase):

    def test_series_instance(self):
        s = expd.ModelSeries([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        self.assertTrue(isinstance(s, expd.ModelSeries))

        s = s[['A', 'B']]
        self.assertEqual(len(s), 2)
        self.assertTrue(isinstance(s, expd.ModelSeries))

    def test_series_to_frame(self):
        s = expd.ModelSeries([1, 2, 3, 4, 5])
        self.assertTrue(isinstance(s, expd.ModelSeries))

        df = s.to_frame()
        self.assertTrue(isinstance(df, expd.ModelFrame))
        self.assert_index_equal(df.columns, pd.Index([0]))

        df = s.to_frame(name='x')
        self.assertTrue(isinstance(df, expd.ModelFrame))
        self.assert_index_equal(df.columns, pd.Index(['x']))

        s = expd.ModelSeries([1, 2, 3, 4, 5], name='name')
        self.assertTrue(isinstance(s, expd.ModelSeries))

        df = s.to_frame()
        self.assertTrue(isinstance(df, expd.ModelFrame))
        self.assert_index_equal(df.columns, pd.Index(['name']))

        df = s.to_frame(name='x')
        self.assertTrue(isinstance(df, expd.ModelFrame))
        self.assert_index_equal(df.columns, pd.Index(['x']))

    def test_preprocessing_normalize(self):
        s = expd.ModelSeries([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        self.assertTrue(isinstance(s, expd.ModelSeries))
        result = s.preprocessing.normalize()
        expected = pp.normalize(np.atleast_2d(s.values))[0, :]

        self.assertTrue(isinstance(result, expd.ModelSeries))
        self.assert_numpy_array_almost_equal(result.values, expected)
        self.assert_index_equal(result.index, s.index)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
