#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.cross_decomposition as cd

import expandas as expd
import expandas.util.testing as tm


class TestCrossDecomposition(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.cross_decomposition.PLSRegression, cd.PLSRegression)
        self.assertIs(df.cross_decomposition.PLSCanonical, cd.PLSCanonical)
        self.assertIs(df.cross_decomposition.CCA, cd.CCA)
        self.assertIs(df.cross_decomposition.PLSSVD, cd.PLSSVD)

    def test_PLSCannonical(self):
        n = 500
        np.random.seed(1)

        # 2 latents vars:
        l1 = np.random.normal(size=n)
        l2 = np.random.normal(size=n)

        latents = np.array([l1, l1, l2, l2]).T
        X = latents + np.random.normal(size=4 * n).reshape((n, 4))
        Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

        X_train = X[:n / 2]
        Y_train = Y[:n / 2]
        X_test = X[n / 2:]
        Y_test = Y[n / 2:]

        train = expd.ModelFrame(X_train, target=Y_train)
        test = expd.ModelFrame(X_test, target=Y_test)
        self.assertTrue(train.has_target())
        expected = pd.Index(['.target 0', '.target 1', '.target 2', '.target 3'])
        self.assert_index_equal(train.target_name, expected)
        self.assertEqual(train.data.shape, X_train.shape)
        self.assertEqual(train.target.shape, Y_train.shape)

        plsca1 = train.cross_decomposition.PLSCanonical(n_components=2)
        plsca2 = cd.PLSCanonical(n_components=2)

        train.fit(plsca1)
        plsca2.fit(X_train, Y_train)

        result_tr = train.transform(plsca1)
        result_test = test.transform(plsca1)

        expected_tr = plsca2.transform(X_train, Y_train)
        expected_test = plsca2.transform(X_test, Y_test)

        self.assertTrue(isinstance(result_tr, expd.ModelFrame))
        self.assertTrue(isinstance(result_test, expd.ModelFrame))
        self.assert_numpy_array_equal(result_tr.data.values, expected_tr[0])
        self.assert_numpy_array_equal(result_tr.target.values, expected_tr[1])
        self.assert_numpy_array_equal(result_test.data.values, expected_test[0])
        self.assert_numpy_array_equal(result_test.target.values, expected_test[1])

    def test_CCA(self):
        n = 500
        np.random.seed(1)

        # 2 latents vars:
        l1 = np.random.normal(size=n)
        l2 = np.random.normal(size=n)

        latents = np.array([l1, l1, l2, l2]).T
        X = latents + np.random.normal(size=4 * n).reshape((n, 4))
        Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

        X_train = X[:n / 2]
        Y_train = Y[:n / 2]
        X_test = X[n / 2:]
        Y_test = Y[n / 2:]

        train = expd.ModelFrame(X_train, target=Y_train)
        test = expd.ModelFrame(X_test, target=Y_test)
        self.assertTrue(train.has_target())
        expected = pd.Index(['.target 0', '.target 1', '.target 2', '.target 3'])
        self.assert_index_equal(train.target_name, expected)
        self.assertEqual(train.data.shape, X_train.shape)
        self.assertEqual(train.target.shape, Y_train.shape)

        cca1 = train.cross_decomposition.CCA(n_components=2)
        cca2 = cd.CCA(n_components=2)

        train.fit(cca1)
        cca2.fit(X_train, Y_train)

        result_tr = train.transform(cca1)
        result_test = test.transform(cca1)

        expected_tr = cca2.transform(X_train, Y_train)
        expected_test = cca2.transform(X_test, Y_test)

        self.assertTrue(isinstance(result_tr, expd.ModelFrame))
        self.assertTrue(isinstance(result_test, expd.ModelFrame))
        self.assert_numpy_array_equal(result_tr.data.values, expected_tr[0])
        self.assert_numpy_array_equal(result_tr.target.values, expected_tr[1])
        self.assert_numpy_array_equal(result_test.data.values, expected_test[0])
        self.assert_numpy_array_equal(result_test.target.values, expected_test[1])

    def test_PLSRegression(self):

        n = 1000
        q = 3
        p = 10
        X = np.random.normal(size=n * p).reshape((n, p))
        B = np.array([[1, 2] + [0] * (p - 2)] * q).T
        # each Yj = 1*X1 + 2*X2 + noize
        Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5

        df = expd.ModelFrame(X, target=Y)
        pls1 = df.cross_decomposition.PLSRegression(n_components=3)
        df.fit(pls1)
        result = df.predict(pls1)

        pls2 = cd.PLSRegression(n_components=3)
        pls2.fit(X, Y)
        expected = pls2.predict(X)

        self.assertTrue(isinstance(result, expd.ModelFrame))
        self.assert_numpy_array_almost_equal(result.values, expected)

"""
###############################################################################
# CCA (PLS mode B with symmetric deflation)

cca = CCA(n_components=2)
cca.fit(X_train, Y_train)
X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
X_test_r, Y_test_r = plsca.transform(X_test, Y_test)
"""

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
