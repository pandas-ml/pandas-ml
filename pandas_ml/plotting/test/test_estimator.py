#!/usr/bin/env python

import sklearn.datasets as datasets

import pandas_ml as pdml
import pandas_ml.util.testing as tm

import matplotlib
matplotlib.use('Agg')


class TestPlotting(tm.PlottingTestCase):

    def test_no_estimator(self):
        df = pdml.ModelFrame(datasets.load_iris())
        with tm.assertRaises(ValueError):
            df.plot_estimator()

    def test_not_supported_estimator(self):
        df = pdml.ModelFrame(datasets.load_iris())
        df.fit(df.cluster.KMeans(n_clusters=3))

        with tm.assertRaises(NotImplementedError):
            df.plot_estimator()

    def test_regression_plot_2d(self):
        df = pdml.ModelFrame(datasets.load_diabetes())
        df.data = df.data[[0]]
        df.fit(df.linear_model.LinearRegression())
        ax = df.plot_estimator()
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_regression_plot_3d(self):
        df = pdml.ModelFrame(datasets.load_diabetes())
        df.data = df.data[[0, 2]]
        df.fit(df.linear_model.LinearRegression())
        ax = df.plot_estimator()

        from mpl_toolkits.mplot3d import Axes3D
        self.assertIsInstance(ax, Axes3D)

    def test_classification_plot_proba(self):
        df = pdml.ModelFrame(datasets.load_iris())
        df.data = df.data.iloc[:, [0, 1]]
        df.fit(df.svm.SVC(C=1.0, probability=True))
        axes = df.plot_estimator()
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))

    def test_classification_plot_decision(self):
        df = pdml.ModelFrame(datasets.load_iris())
        df.data = df.data.iloc[:, [0, 1]]
        df.fit(df.svm.SVC(C=1.0))
        axes = df.plot_estimator()
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))

    def test_classification_plot_proba_highdim(self):
        df = pdml.ModelFrame(datasets.load_iris())
        df.fit(df.svm.SVC(C=1.0, probability=True))
        axes = df.plot_estimator()
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))

    def test_classification_plot_decision_highdim(self):
        df = pdml.ModelFrame(datasets.load_iris())
        df.fit(df.svm.SVC(C=1.0))
        axes = df.plot_estimator()
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
