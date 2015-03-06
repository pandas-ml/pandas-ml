#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.svm as svm

import expandas as expd
import expandas.util.testing as tm


class TestSVM(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.svm.SVC, svm.SVC)
        self.assertIs(df.svm.LinearSVC, svm.LinearSVC)
        self.assertIs(df.svm.NuSVC, svm.NuSVC)
        self.assertIs(df.svm.SVR, svm.SVR)
        self.assertIs(df.svm.NuSVR, svm.NuSVR)
        self.assertIs(df.svm.OneClassSVM, svm.OneClassSVM)

    def test_l1_min_c(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        result = df.svm.l1_min_c()
        expected = svm.l1_min_c(iris.data, iris.target)
        self.assertAlmostEqual(result, expected)

    def test_Regressions(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        models = ['SVR', 'NuSVR']
        for model in models:
            mod1 = getattr(df.svm, model)(random_state=self.random_state)
            mod2 = getattr(svm, model)(random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertTrue(isinstance(result, expd.ModelSeries))
            self.assert_numpy_array_almost_equal(result.values, expected)

            self.assertTrue(isinstance(df.predicted, expd.ModelSeries))
            self.assert_numpy_array_almost_equal(df.predicted.values, expected)

    def test_Classifications(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        models = ['LinearSVC', 'NuSVC']
        for model in models:
            mod1 = getattr(df.svm, model)(random_state=self.random_state)
            mod2 = getattr(svm, model)(random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertTrue(isinstance(result, expd.ModelSeries))
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
