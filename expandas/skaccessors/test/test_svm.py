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

    def test_predict_proba(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        models = ['SVC']
        for model in models:
            mod1 = getattr(df.svm, model)(probability=True, random_state=self.random_state)
            mod2 = getattr(svm, model)(probability=True, random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertTrue(isinstance(result, expd.ModelSeries))
            self.assert_numpy_array_almost_equal(result.values, expected)

            result = df.predict_proba(mod1)
            expected = mod2.predict_proba(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

            self.assert_numpy_array_almost_equal(df.proba.values, expected)

            result = df.predict_log_proba(mod1)
            expected = mod2.predict_log_proba(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

            self.assert_numpy_array_almost_equal(df.log_proba.values, expected)

    def test_predict_automatic(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        models = ['SVC']
        for model in models:
            df = expd.ModelFrame(iris)
            mod1 = getattr(df.svm, model)(probability=True, random_state=self.random_state)
            mod2 = getattr(svm, model)(probability=True, random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            # test automatically calls related methods
            with tm.assert_produces_warning(UserWarning):
                result = df.predicted
            expected = mod2.predict(iris.data)

            self.assertTrue(isinstance(result, expd.ModelSeries))
            self.assert_numpy_array_almost_equal(result.values, expected)

            with tm.assert_produces_warning(UserWarning):
                result = df.proba
            expected = mod2.predict_proba(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

            with tm.assert_produces_warning(UserWarning):
                result = df.log_proba
            expected = mod2.predict_log_proba(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
