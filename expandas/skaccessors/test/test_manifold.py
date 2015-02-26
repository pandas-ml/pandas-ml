#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.manifold as manifold

import expandas as expd
import expandas.util.testing as tm


class TestManifold(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.manifold.LocallyLinearEmbedding, manifold.LocallyLinearEmbedding)
        self.assertIs(df.manifold.Isomap, manifold.Isomap)
        self.assertIs(df.manifold.MDS, manifold.MDS)
        self.assertIs(df.manifold.SpectralEmbedding, manifold.SpectralEmbedding)
        self.assertIs(df.manifold.TSNE, manifold.TSNE)

    def test_locally_linear_embedding(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        result = df.manifold.locally_linear_embedding(3, 3)
        expected = manifold.locally_linear_embedding(iris.data, 3, 3)

        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result[0], expd.ModelFrame))
        self.assert_index_equal(result[0].index, df.index)
        self.assert_numpy_array_equal(result[0].values, expected[0])

        self.assertEqual(result[1], expected[1])

    def test_Isomap(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        models = ['Isomap']
        for model in models:
            mod1 = getattr(df.manifold, model)() # random_state=self.random_state)
            mod2 = getattr(manifold, model)() #random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data)

            result = df.transform(mod1)
            expected = mod2.transform(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.data.values, expected)

    def test_MDS(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        models = ['MDS']
        for model in models:
            mod1 = getattr(df.manifold, model)(random_state=self.random_state)
            mod2 = getattr(manifold, model)(random_state=self.random_state)

            result = df.fit_transform(mod1)
            expected = mod2.fit_transform(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.data.values, expected)

    def test_TSNE(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        models = ['TSNE']
        for model in models:
            mod1 = getattr(df.manifold, model)(random_state=self.random_state, init='pca')
            mod2 = getattr(manifold, model)(random_state=self.random_state, init='pca')

            np.random.seed(1)
            result = df.fit_transform(mod1)
            np.random.seed(1)
            expected = mod2.fit_transform(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.data.shape, expected.shape)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
