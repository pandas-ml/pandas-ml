#!/usr/bin/env python

import numpy as np
import pandas as pd

import sklearn.datasets as datasets

import pandas_ml as pdml
import pandas_ml.util.testing as tm

import matplotlib
matplotlib.use('Agg')


import seaborn as sns


class TestSeabornBasic(tm.PlottingTestCase):

    def setUp(self):
        self.iris = pdml.ModelFrame(datasets.load_iris())

        self.diabetes = pdml.ModelFrame(datasets.load_diabetes())
        # convert columns to str
        self.diabetes.columns = ['col{0}'.format(c) if isinstance(c, int) else c for c in self.diabetes.columns]

    def test_jointplot(self):
        df = self.iris

        jg = df.sns.jointplot(df.columns[1])
        self.assertIsInstance(jg, sns.JointGrid)
        self.assertEqual(jg.ax_joint.get_xlabel(), df.columns[1])
        self.assertEqual(jg.ax_joint.get_ylabel(), '.target')

        jg = df.sns.jointplot(df.columns[2], df.columns[3])
        self.assertIsInstance(jg, sns.JointGrid)
        self.assertEqual(jg.ax_joint.get_xlabel(), df.columns[2])
        self.assertEqual(jg.ax_joint.get_ylabel(), df.columns[3])

    def test_pairplot(self):
        df = self.iris

        pg = df.sns.pairplot()
        self._check_axes_shape(pg.axes, axes_num=25, layout=(5, 5), figsize=None)
        for i in range(5):
            self.assertEqual(pg.axes[i][0].get_ylabel(), df.columns[i])
            self.assertEqual(pg.axes[-1][i].get_xlabel(), df.columns[i])

    def test_distplot(self):
        return # ToDo: only fails on Travis

        df = self.iris

        ax = df.sns.distplot()
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), '.target')

        # pass scalar (str)
        ax = df.sns.distplot(df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), df.columns[1])

        # pass Series
        ax = df.sns.distplot(df[df.columns[2]])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), df.columns[2])

    def test_kdeplot(self):
        pass

    def test_rugplot(self):
        df = self.iris

        ax = df.sns.rugplot()
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        # rugplot does not add label

        # pass scalar (str)
        ax = df.sns.rugplot(df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)

        # pass Series
        ax = df.sns.rugplot(df[df.columns[2]])
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    # Regression

    def test_lmplot(self):
        df = self.diabetes

        fg = df.sns.lmplot('col2')
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), 'col2')
        self.assertEqual(fg.ax.get_ylabel(), '.target')

        fg = df.sns.lmplot('col2', 'col3')
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), 'col2')
        self.assertEqual(fg.ax.get_ylabel(), 'col3')


    def test_regression_plot(self):
        df = self.diabetes

        plots = ['regplot', 'residplot']

        for plot in plots:
            func = getattr(df.sns, plot)

            ax = func('col2')
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), 'col2')
            self.assertEqual(ax.get_ylabel(), '.target')

            ax = func('col2', 'col3')
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), 'col2')
            self.assertEqual(ax.get_ylabel(), 'col3')

    def test_interactplot(self):
        pass

    def test_coefplot(self):
        pass

    # Categorical

    def test_factorplots(self):
        df = self.iris

        fg = df.sns.factorplot(df.columns[1])
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), '.target')
        self.assertEqual(fg.ax.get_ylabel(), df.columns[1])

        fg = df.sns.factorplot(x=df.columns[1])
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), df.columns[1])
        self.assertEqual(fg.ax.get_ylabel(), '.target')

        fg = df.sns.factorplot(x=df.columns[1], y=df.columns[2])
        self.assertIsInstance(fg, sns.FacetGrid)
        self.assertEqual(fg.ax.get_xlabel(), df.columns[1])
        self.assertEqual(fg.ax.get_ylabel(), df.columns[2])

    def test_categoricalplots(self):
        df = self.iris

        plots = ['boxplot', 'violinplot', 'stripplot']

        for plot in plots:
            func = getattr(df.sns, plot)
            ax = func(df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), '.target')
            self.assertEqual(ax.get_ylabel(), df.columns[1])

            ax = func(y=df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), '.target')
            self.assertEqual(ax.get_ylabel(), df.columns[1])

            ax = func(x=df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), df.columns[1])
            self.assertEqual(ax.get_ylabel(), '.target')

            ax = func(x=df.columns[1], y=df.columns[2])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), df.columns[1])
            self.assertEqual(ax.get_ylabel(), df.columns[2])

    def test_categorical_mean_plots(self):
        df = self.iris

        plots = ['pointplot', 'barplot']

        for plot in plots:
            func = getattr(df.sns, plot)
            ax = func(df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), '.target')
            self.assertEqual(ax.get_ylabel(), 'mean({0})'.format(df.columns[1]))

            ax = func(y=df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), '.target')
            self.assertEqual(ax.get_ylabel(), 'mean({0})'.format(df.columns[1]))

            ax = func(x=df.columns[1])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), df.columns[1])
            self.assertEqual(ax.get_ylabel(), 'mean({0})'.format('.target'))

            ax = func(x=df.columns[1], y=df.columns[2])
            self.assertIsInstance(ax, matplotlib.axes.Axes)
            self.assertEqual(ax.get_xlabel(), df.columns[1])
            self.assertEqual(ax.get_ylabel(), 'mean({0})'.format(df.columns[2]))

    def test_count_plots(self):
        df = self.iris

        ax = df.sns.countplot()
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), '.target')
        self.assertEqual(ax.get_ylabel(), 'count')

        return # ToDo: only fails on Travis

        ax = df.sns.countplot(df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), df.columns[1])
        self.assertEqual(ax.get_ylabel(), 'count')

        ax = df.sns.countplot(x=df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), df.columns[1])
        self.assertEqual(ax.get_ylabel(), 'count')

        ax = df.sns.countplot(y=df.columns[1])
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_xlabel(), 'count')
        self.assertEqual(ax.get_ylabel(), df.columns[1])

        with tm.assertRaises(TypeError):
            # can't pass both x and y
            df.sns.countplot(x=df.columns[1], y=df.columns[2])

    # Matrix

    def test_heatmap(self):
        pass

    def test_clustermap(self):
        pass

    # Timeseries

    def test_tsplot(self):
        pass

    # AxisGrid

    def test_facetgrid(self):
        df = self.iris

        fg = df.sns.FacetGrid(df.columns[0])
        self.assertIsInstance(fg, sns.FacetGrid)
        self._check_axes_shape(fg.axes, axes_num=3, layout=(3, 1), figsize=None)

        fg = df.sns.FacetGrid(row=df.columns[0])
        self.assertIsInstance(fg, sns.FacetGrid)
        self._check_axes_shape(fg.axes, axes_num=3, layout=(3, 1), figsize=None)

        fg = df.sns.FacetGrid(col=df.columns[0])
        self.assertIsInstance(fg, sns.FacetGrid)
        self._check_axes_shape(fg.axes, axes_num=3, layout=(1, 3), figsize=None)

    def test_pairgrid(self):
        df = self.iris

        pg = df.sns.PairGrid()
        self.assertIsInstance(pg, sns.PairGrid)
        self._check_axes_shape(pg.axes, axes_num=25, layout=(5, 5), figsize=None)

    def test_jointgrid(self):
        df = self.iris

        jg = df.sns.JointGrid(x=df.columns[1], y=df.columns[1])
        self.assertIsInstance(jg, sns.JointGrid)



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
