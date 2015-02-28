#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.linear_model as lm

import expandas as expd
import expandas.util.testing as tm


class TestLinearModel(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.linear_model.ARDRegression, lm.ARDRegression)
        self.assertIs(df.linear_model.BayesianRidge, lm.BayesianRidge)
        self.assertIs(df.linear_model.ElasticNet, lm.ElasticNet)
        self.assertIs(df.linear_model.ElasticNetCV, lm.ElasticNetCV)
        self.assertIs(df.linear_model.Lars, lm.Lars)
        self.assertIs(df.linear_model.LarsCV, lm.LarsCV)
        self.assertIs(df.linear_model.Lasso, lm.Lasso)
        self.assertIs(df.linear_model.LassoCV, lm.LassoCV)
        self.assertIs(df.linear_model.LassoLars, lm.LassoLars)
        self.assertIs(df.linear_model.LassoLarsCV, lm.LassoLarsCV)
        self.assertIs(df.linear_model.LassoLarsIC, lm.LassoLarsIC)
        self.assertIs(df.linear_model.LinearRegression, lm.LinearRegression)
        self.assertIs(df.linear_model.LogisticRegression, lm.LogisticRegression)
        self.assertIs(df.linear_model.MultiTaskLasso, lm.MultiTaskLasso)
        self.assertIs(df.linear_model.MultiTaskElasticNet, lm.MultiTaskElasticNet)
        self.assertIs(df.linear_model.MultiTaskLassoCV, lm.MultiTaskLassoCV)
        self.assertIs(df.linear_model.MultiTaskElasticNetCV, lm.MultiTaskElasticNetCV)
        self.assertIs(df.linear_model.OrthogonalMatchingPursuit, lm.OrthogonalMatchingPursuit)
        self.assertIs(df.linear_model.OrthogonalMatchingPursuitCV, lm.OrthogonalMatchingPursuitCV)
        self.assertIs(df.linear_model.PassiveAggressiveClassifier, lm.PassiveAggressiveClassifier)
        self.assertIs(df.linear_model.PassiveAggressiveRegressor, lm.PassiveAggressiveRegressor)
        self.assertIs(df.linear_model.Perceptron, lm.Perceptron)
        self.assertIs(df.linear_model.RandomizedLasso, lm.RandomizedLasso)
        self.assertIs(df.linear_model.RandomizedLogisticRegression, lm.RandomizedLogisticRegression)
        self.assertIs(df.linear_model.RANSACRegressor, lm.RANSACRegressor)
        self.assertIs(df.linear_model.Ridge, lm.Ridge)
        self.assertIs(df.linear_model.RidgeClassifier, lm.RidgeClassifier)
        self.assertIs(df.linear_model.RidgeClassifierCV, lm.RidgeClassifierCV)
        self.assertIs(df.linear_model.RidgeCV, lm.RidgeCV)
        self.assertIs(df.linear_model.SGDClassifier, lm.SGDClassifier)
        self.assertIs(df.linear_model.SGDRegressor, lm.SGDRegressor)

    def test_lars_path(self):
        diabetes = datasets.load_diabetes()
        df = expd.ModelFrame(diabetes)

        result = df.linear_model.lars_path()
        expected = lm.lars_path(diabetes.data, diabetes.target)

        self.assertEqual(len(result), 3)
        self.assert_numpy_array_equal(result[0], expected[0])
        self.assert_numpy_array_equal(result[1], expected[1])
        self.assertTrue(isinstance(result[2], expd.ModelFrame))
        self.assert_index_equal(result[2].index, df.data.columns)
        self.assert_numpy_array_equal(result[2].values, expected[2])

    def test_lasso_path(self):
        diabetes = datasets.load_diabetes()
        df = expd.ModelFrame(diabetes)

        result = df.linear_model.lasso_path()
        expected = lm.lasso_path(diabetes.data, diabetes.target)

        self.assertEqual(len(result), 3)
        self.assert_numpy_array_equal(result[0], expected[0])
        self.assertTrue(isinstance(result[1], expd.ModelFrame))
        self.assert_index_equal(result[1].index, df.data.columns)
        self.assert_numpy_array_equal(result[1].values, expected[1])
        self.assert_numpy_array_equal(result[2], expected[2])

        result = df.linear_model.lasso_path(return_models=True)
        expected = lm.lasso_path(diabetes.data, diabetes.target, return_models=True)

        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0], lm.ElasticNet))
        self.assertEqual(len(result), len(expected))

    def test_lasso_stability_path(self):
        diabetes = datasets.load_diabetes()
        df = expd.ModelFrame(diabetes)

        result = df.linear_model.lasso_stability_path(random_state=self.random_state)
        expected = lm.lasso_stability_path(diabetes.data, diabetes.target,
                                           random_state=self.random_state)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_equal(result[0], expected[0])

        self.assertTrue(isinstance(result[1], expd.ModelFrame))
        self.assert_index_equal(result[1].index, df.data.columns)
        self.assert_numpy_array_equal(result[1].values, expected[1])

    def test_orthogonal_mp(self):
        diabetes = datasets.load_diabetes()
        df = expd.ModelFrame(diabetes)

        result = df.linear_model.orthogonal_mp()
        expected = lm.orthogonal_mp(diabetes.data, diabetes.target)
        self.assert_numpy_array_equal(result, expected)

        result = df.linear_model.orthogonal_mp(return_path=True)
        expected = lm.orthogonal_mp(diabetes.data, diabetes.target, return_path=True)
        self.assert_numpy_array_equal(result, expected)

    def test_orthogonal_mp_gram(self):
        diabetes = datasets.load_diabetes()
        df = expd.ModelFrame(diabetes)

        result = df.linear_model.orthogonal_mp()

        gram = diabetes.data.T.dot(diabetes.data)
        Xy = diabetes.data.T.dot(diabetes.target)
        expected = lm.orthogonal_mp_gram(gram, Xy)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_Regresions(self):
        diabetes = datasets.load_diabetes()
        df = expd.ModelFrame(diabetes)

        models = ['LinearRegression', 'Ridge', 'RidgeCV', 'Lasso']
        for model in models:
            mod1 = getattr(df.linear_model, model)()
            mod2 = getattr(lm, model)()

            df.fit(mod1)
            mod2.fit(diabetes.data, diabetes.target)

            result = df.predict(mod1)
            expected = mod2.predict(diabetes.data)

            self.assertTrue(isinstance(result, expd.ModelSeries))
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
