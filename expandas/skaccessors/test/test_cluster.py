#!/usr/bin/env python

from __future__ import unicode_literals

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.cluster as cluster

import expandas as expd
import expandas.util.testing as tm


class TestMetrics(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.cluster.AffinityPropagation, cluster.AffinityPropagation)
        self.assertIs(df.cluster.AgglomerativeClustering, cluster.AgglomerativeClustering)
        self.assertIs(df.cluster.DBSCAN, cluster.DBSCAN)
        self.assertIs(df.cluster.FeatureAgglomeration, cluster.FeatureAgglomeration)
        self.assertIs(df.cluster.KMeans, cluster.KMeans)
        self.assertIs(df.cluster.MiniBatchKMeans, cluster.MiniBatchKMeans)
        self.assertIs(df.cluster.MeanShift, cluster.MeanShift)
        self.assertIs(df.cluster.SpectralClustering, cluster.SpectralClustering)
        self.assertIs(df.cluster.Ward, cluster.Ward)

    def test_estimate_bandwidth(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        result = df.cluster.estimate_bandwidth(random_state=self.random_state)
        expected = cluster.estimate_bandwidth(iris.data, random_state=self.random_state)
        self.assertEqual(result, expected)

    def test_k_means(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        result = df.cluster.k_means(3, random_state=self.random_state)
        expected = cluster.k_means(iris.data, 3, random_state=self.random_state)

        self.assertEqual(len(result), 3)
        self.assert_numpy_array_almost_equal(result[0], expected[0])

        self.assertTrue(isinstance(result[1], pd.Series))
        self.assert_index_equal(result[1].index, df.index)
        self.assert_numpy_array_equal(result[1].values, expected[1])

        self.assertAlmostEqual(result[2], expected[2])

    def test_ward_tree(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        result = df.cluster.ward_tree()
        expected = cluster.ward_tree(iris.data)

        self.assertEqual(len(result), 4)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertEqual(result[2], expected[2])
        self.assertEqual(result[3], expected[3])

        connectivity = np.ones((len(df), len(df)))
        result = df.cluster.ward_tree(connectivity)
        expected = cluster.ward_tree(iris.data, connectivity)

        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertEqual(result[2], expected[2])
        self.assert_numpy_array_almost_equal(result[3], expected[3])

    def test_affinity_propagation(self):
        iris = datasets.load_iris()
        similality = np.cov(iris.data)
        df = expd.ModelFrame(similality)

        result = df.cluster.affinity_propagation()
        expected = cluster.affinity_propagation(similality)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])

        self.assertTrue(isinstance(result[1], pd.Series))
        self.assert_index_equal(result[1].index, df.index)
        self.assert_numpy_array_equal(result[1].values, expected[1])

    def test_affinity_propagation_class(self):
        from sklearn.datasets.samples_generator import make_blobs

        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=300, centers=centers,
                                    cluster_std=0.5,random_state=0)

        df = expd.ModelFrame(data=X, target=labels_true)
        af = df.cluster.AffinityPropagation(preference=-50)
        df.fit(af)

        af2 = cluster.AffinityPropagation(preference=-50).fit(X)

        self.assert_numpy_array_equal(af.cluster_centers_indices_,
                                      af2.cluster_centers_indices_)
        self.assert_numpy_array_equal(af.labels_,
                                      af2.labels_)

    def test_dbscan(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        result = df.cluster.dbscan(random_state=self.random_state)
        expected = cluster.dbscan(iris.data, random_state=self.random_state)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assertTrue(isinstance(result[1], pd.Series))
        self.assert_index_equal(result[1].index, df.index)
        self.assert_numpy_array_equal(result[1].values, expected[1])

    def test_mean_shift(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        result = df.cluster.mean_shift()
        expected = cluster.mean_shift(iris.data)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assertTrue(isinstance(result[1], pd.Series))
        self.assert_index_equal(result[1].index, df.index)
        self.assert_numpy_array_equal(result[1].values, expected[1])

    def test_spectral_clustering(self):
        data = np.random.randn(100)
        # create affinity matrix
        affinity = np.subtract.outer(data, data)
        affinity = np.abs(affinity)
        df = expd.ModelFrame(affinity)

        result = df.cluster.spectral_clustering(random_state=self.random_state)
        expected = cluster.spectral_clustering(affinity, random_state=self.random_state)

        self.assertTrue(isinstance(result, pd.Series))
        self.assert_index_equal(result.index, df.index)
        self.assert_numpy_array_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
