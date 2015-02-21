#!/usr/bin/env python

from __future__ import unicode_literals

import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets

import expandas as expd
import expandas.util.testing as tm


class TestModelFrame(tm.TestCase):

    def test_frame_instance(self):

        df = expd.ModelFrame(datasets.load_digits())
        self.assertTrue(isinstance(df, expd.ModelFrame))

        train_df, test_df = df.cross_validation.train_test_split()

        self.assertTrue(isinstance(train_df, expd.ModelFrame))
        self.assertTrue(isinstance(test_df, expd.ModelFrame))
        self.assertTrue(isinstance(train_df.iloc[:, 2:3], expd.ModelFrame))

    def test_frame_slice(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        expected = ['.target'] + iris.feature_names
        self.assertEqual(df.columns.tolist(), expected)

        s = df['.target']
        self.assertTrue(isinstance(s, expd.ModelSeries))

        s = df[['.target']]
        self.assertTrue(isinstance(s, expd.ModelFrame))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
