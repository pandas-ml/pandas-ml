#!/usr/bin/env python

import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.metrics as metrics

import expandas as expd
import expandas.util.testing as tm


class TestMetricsCommon(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.metrics.make_scorer, metrics.make_scorer)

    def test_make_scorer(self):
        df = expd.ModelFrame([])

        result = df.metrics.make_scorer(metrics.fbeta_score, beta=2)
        expected = metrics.make_scorer(metrics.fbeta_score, beta=2)

        self.assertEqual(result._kwargs, expected._kwargs)
        self.assertEqual(result._sign, expected._sign)
        self.assertEqual(result._score_func, expected._score_func)


class TestClassificationMetrics(tm.TestCase):

    def setUp(self):
        import sklearn.svm as svm
        self.digits = datasets.load_digits()

        self.df = expd.ModelFrame(self.digits)

        estimator1 = svm.LinearSVC(C=1.0, random_state=self.random_state)
        self.df.fit(estimator1)
        self.df.predict(estimator1)

        estimator2 = svm.LinearSVC(C=1.0, random_state=self.random_state)
        estimator2.fit(self.digits.data, self.digits.target)
        self.digits_pred = estimator2.predict(self.digits.data)

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.metrics.make_scorer, metrics.make_scorer)

    def test_make_scorer(self):
        df = expd.ModelFrame([])

        result = df.metrics.make_scorer(metrics.fbeta_score, beta=2)
        expected = metrics.make_scorer(metrics.fbeta_score, beta=2)

        self.assertEqual(result._kwargs, expected._kwargs)
        self.assertEqual(result._sign, expected._sign)
        self.assertEqual(result._score_func, expected._score_func)

    def test_accuracy_score(self):
        result = self.df.metrics.accuracy_score()
        expected = metrics.accuracy_score(self.digits.target, self.digits_pred)
        self.assertEqual(result, expected)

        result = self.df.metrics.accuracy_score(normalize=False)
        expected = metrics.accuracy_score(self.digits.target,
                                          self.digits_pred,
                                          normalize=False)
        self.assertEqual(result, expected)

    def test_auc(self):
        # result = self.df.metrics.auc(reorder=True)
        # expected = metrics.auc(self.digits.data, self.digits.target, reorder=True)
        # self.assertEqual(result, expected)
        pass

    def test_average_precision_score(self):
        pass

    def test_classification_report(self):
        result = self.df.metrics.classification_report()
        expected = metrics.classification_report(self.digits.target,
                                                 self.digits_pred)
        self.assertEqual(result, expected)

    def test_confusion_matrix(self):
        result = self.df.metrics.confusion_matrix()
        expected = metrics.confusion_matrix(self.digits.target,
                                            self.digits_pred)

        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assert_numpy_array_equal(result.values, expected)

    def test_f1_score(self):
        result = self.df.metrics.f1_score()
        expected = metrics.f1_score(self.digits.target, self.digits_pred)
        self.assertEqual(result, expected)

    def test_fbeta_score(self):
        result = self.df.metrics.fbeta_score(beta=0.5)
        expected = metrics.fbeta_score(self.digits.target, self.digits_pred,
                                       beta=0.5)
        self.assertEqual(result, expected)

        result = self.df.metrics.fbeta_score(beta=0.5, average='macro')
        expected = metrics.fbeta_score(self.digits.target, self.digits_pred,
                                       beta=0.5, average='macro')
        self.assertEqual(result, expected)

    def test_hamming_loss(self):
        result = self.df.metrics.hamming_loss()
        expected = metrics.hamming_loss(self.digits.target, self.digits_pred)
        self.assertEqual(result, expected)

    def test_hinge_loss(self):
        pass

    def test_jaccard_similarity_score(self):
        result = self.df.metrics.jaccard_similarity_score()
        expected = metrics.jaccard_similarity_score(self.digits.target,
                                                    self.digits_pred)
        self.assertEqual(result, expected)

    def test_log_loss(self):
        pass

    def test_matthews_corrcoef(self):
        pass
        # ValueError: multiclass is not supported

        """
        result = self.df.metrics.matthews_corrcoef()
        expected = metrics.matthews_corrcoef(self.digits.target, self.digits_pred)
        self.assertEqual(result, expected)
        """

    def test_precision_recall_curve(self):
        pass

    def test_precision_recall_fscore_support(self):
        pass

    def test_precision_score(self):
        result = self.df.metrics.precision_score()
        expected = metrics.precision_score(self.digits.target, self.digits_pred)
        self.assertEqual(result, expected)

    def test_recall_score(self):
        result = self.df.metrics.recall_score()
        expected = metrics.recall_score(self.digits.target, self.digits_pred)
        self.assertEqual(result, expected)

    def test_roc_auc_score(self):
        pass

    def test_roc_curve(self):
        pass

    def test_zero_one_loss(self):
        result = self.df.metrics.zero_one_loss()
        expected = metrics.zero_one_loss(self.digits.target, self.digits_pred)
        self.assertEqual(result, expected)


class TestRegressionMetrics(tm.TestCase):

    def setUp(self):
        from sklearn import linear_model
        data = [[0, 0], [1, 1], [2, 2]]
        self.target = [0, 1, 2]

        self.df = expd.ModelFrame(data=data, target=self.target)

        estimator1 = linear_model.LinearRegression()
        self.df.fit(estimator1)
        self.df.predict(estimator1)

        estimator2 = linear_model.LinearRegression()

        estimator2.fit(data, self.target)
        self.pred = estimator2.predict(data)

    def test_explained_variance_score(self):
        result = self.df.explained_variance_score()
        expected = metrics.explained_variance_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_mean_absolute_error(self):
        result = self.df.metrics.mean_absolute_error()
        expected = metrics.mean_absolute_error(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_mean_squared_error(self):
        result = self.df.metrics.mean_squared_error()
        expected = metrics.mean_squared_error(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_explained_variance_score(self):
        result = self.df.metrics.explained_variance_score()
        expected = metrics.explained_variance_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_r2_score(self):
        result = self.df.metrics.r2_score()
        expected = metrics.r2_score(self.target, self.pred)
        self.assertEqual(result, expected)


class Test1ClusteringMetrics(tm.TestCase):

    def setUp(self):
        from sklearn import cluster
        iris = datasets.load_iris()
        self.data = iris.data
        self.target = iris.target

        self.df = expd.ModelFrame(iris)
        estimator1 = cluster.KMeans(3, random_state=self.random_state)

        self.df.fit(estimator1)
        self.df.predict(estimator1)

        estimator2 = cluster.KMeans(3, random_state=self.random_state)
        estimator2.fit(iris.data, self.target)
        self.pred = estimator2.predict(iris.data)

    def test_adjusted_mutual_info_score(self):
        result = self.df.metrics.adjusted_mutual_info_score()
        expected = metrics.adjusted_mutual_info_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_adjusted_rand_score(self):
        result = self.df.metrics.adjusted_rand_score()
        expected = metrics.adjusted_rand_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_completeness_score(self):
        result = self.df.metrics.completeness_score()
        expected = metrics.completeness_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_homogeneity_completeness_v_measure(self):
        result = self.df.metrics.homogeneity_completeness_v_measure()
        expected = metrics.homogeneity_completeness_v_measure(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_homogeneity_score(self):
        result = self.df.metrics.homogeneity_score()
        expected = metrics.homogeneity_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_mutual_info_score(self):
        result = self.df.metrics.mutual_info_score()
        expected = metrics.mutual_info_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_normalized_mutual_info_score(self):
        result = self.df.metrics.normalized_mutual_info_score()
        expected = metrics.normalized_mutual_info_score(self.target, self.pred)
        self.assertEqual(result, expected)

    def test_silhouette_score(self):
        result = self.df.metrics.silhouette_score()
        expected = metrics.silhouette_score(self.data, self.pred)
        self.assertAlmostEqual(result, expected)

    def test_silhouette_samples(self):
        result = self.df.metrics.silhouette_samples()
        expected = metrics.silhouette_samples(self.data, self.pred)

        self.assertTrue(isinstance(result, pd.Series))
        self.assert_index_equal(result.index, self.df.index)
        self.assert_numpy_array_almost_equal(result.values, expected)

    def test_v_measure_score(self):
        result = self.df.metrics.v_measure_score()
        expected = metrics.v_measure_score(self.target, self.pred)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
