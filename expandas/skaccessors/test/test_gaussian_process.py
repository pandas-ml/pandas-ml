#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.gaussian_process as gp

import expandas as expd
import expandas.util.testing as tm


class TestGaussianProcess(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        dgp = df.gaussian_process
        self.assertIs(dgp.GaussianProcess, gp.GaussianProcess)
        self.assertIs(dgp.correlation_models.absolute_exponential,
                      gp.correlation_models.absolute_exponential)
        self.assertIs(dgp.correlation_models.squared_exponential,
                      gp.correlation_models.squared_exponential)
        self.assertIs(dgp.correlation_models.generalized_exponential,
                      gp.correlation_models.generalized_exponential)
        self.assertIs(dgp.correlation_models.pure_nugget,
                      gp.correlation_models.pure_nugget)
        self.assertIs(dgp.correlation_models.cubic,
                      gp.correlation_models.cubic)
        self.assertIs(dgp.correlation_models.linear,
                      gp.correlation_models.linear)

    def test_constant(self):
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        df = expd.ModelFrame(X)

        result = df.gaussian_process.regression_models.constant()
        expected = gp.regression_models.constant(X)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_linear(self):
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        df = expd.ModelFrame(X)

        result = df.gaussian_process.regression_models.linear()
        expected = gp.regression_models.linear(X)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_quadratic(self):
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        df = expd.ModelFrame(X)

        result = df.gaussian_process.regression_models.quadratic()
        expected = gp.regression_models.quadratic(X)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_GaussianProcess(self):
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        y = np.sin(X).ravel()
        df = expd.ModelFrame(X, target=y)

        g1 = df.gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        g2 = gp.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)

        g1.fit(X, y)
        g2.fit(X, y)

        x = np.atleast_2d(np.linspace(0, 10, 1000)).T
        tdf = expd.ModelFrame(x)

        y_result, sigma2_result = tdf.predict(g1, eval_MSE=True)
        y_expected, sigma2_expected = g2.predict(x, eval_MSE=True)

        self.assertTrue(isinstance(y_result, expd.ModelSeries))
        self.assert_index_equal(y_result.index, tdf.index)

        self.assertTrue(isinstance(sigma2_result, expd.ModelSeries))
        self.assert_index_equal(sigma2_result.index, tdf.index)

        self.assert_numpy_array_almost_equal(y_result.values, y_expected)
        self.assert_numpy_array_almost_equal(sigma2_result.values, sigma2_expected)

        y_result = tdf.predict(g1)
        y_expected = g2.predict(x)

        self.assertTrue(isinstance(y_result, expd.ModelSeries))
        self.assert_index_equal(y_result.index, tdf.index)

        self.assert_numpy_array_almost_equal(y_result, y_expected)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
