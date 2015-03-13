#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets

import expandas as expd
import expandas.util.testing as tm


class TestDatasets(tm.TestCase):

    def test_boston(self):
        data = datasets.load_boston()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (506, 14))

    def test_diabetes(self):
        data = datasets.load_diabetes()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (442, 11))

    def test_digits(self):
        data = datasets.load_digits()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (1797, 65))

    def test_iris(self):
        data = datasets.load_iris()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (150, 5))

    def test_linnerud(self):
        return
        # data must be 1-dimensional

        # data = datasets.load_linnerud()
        # df = expd.ModelFrame(data)
        # self.assertEqual(df.shape, (150, 5))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
