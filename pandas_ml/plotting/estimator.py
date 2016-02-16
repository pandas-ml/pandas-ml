#!/usr/bin/env python

import numpy as np
import pandas as pd


class _BasePlotter(object):

    def __init__(self, data, estimator):
        self.data = data
        self.estimator = estimator

        assert isinstance(self.data, pd.DataFrame)
        self.n_components = self.data.data.shape[1]

        # used to reduce dimensionality
        from sklearn.decomposition import PCA
        self.transformer = PCA(n_components=2)

    def _get_x_values(self):
        """
        Get data values used as x axis

        Returns
        -------
        x : Series
        """
        assert self.n_components == 1
        return self.data.data.iloc[:, 0]

    def _get_xy_values(self):
        """
        Get data values used as x and y axis

        Returns
        -------
        x, y : tuple of Series
        """
        if self.n_components == 1:
            raise ValueError
        elif self.n_components == 2:
            data = self.data.data
            x = data.iloc[:, 0]
            y = data.iloc[:, 1]
        else:
            self.transformer = self.data.fit(self.transformer)
            data = self.data.transform(self.transformer).data
            x = data.iloc[:, 0]
            y = data.iloc[:, 1]
        return x, y

    def _maybe_inverse(self, data):
        if self.n_components == 2:
            return data
        else:
            return self.transformer.inverse_transform(data)

    @property
    def plt(self):
        import matplotlib.pyplot as plt
        return plt

    def _get_meshgrid(self, xmin, xmax, ymin, ymax, step=100):
        xx = np.linspace(xmin, xmax, step)
        yy = np.linspace(ymin, ymax, step).T
        xx, yy = np.meshgrid(xx, yy)
        return xx, yy

    def _subplots(self, naxes, layout=None):
        """
        Initialize subplots. See pandas.tools.plotting._subplots

        Parameters
        ----------
        naxes : int
            Number of required axes.

        Returns
        -------
        fig, ax : tuple
        """

        return pd.tools.plotting._subplots(naxes=naxes, layout=layout)


class ClassifierPlotter(_BasePlotter):

    def plot(self):
        if hasattr(self.estimator, 'predict_proba'):
            return self._plot_proba()
        else:
            return self._plot_decision()

    def _plot_proba(self):
        target = self.data.target
        n_classes = len(target.unique())
        fig, axes = self._subplots(n_classes)

        x, y = self._get_xy_values()

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        # ToDo: Adjust edges
        xx, yy = self._get_meshgrid(xmin, xmax, ymin, ymax)
        Xfull = np.c_[xx.ravel(), yy.ravel()]
        Xfull = self._maybe_inverse(Xfull)
        probas = self.estimator.predict_proba(Xfull)

        flatten = axes.flatten()

        for k, ax in zip(range(n_classes), flatten):
            ax.imshow(probas[:, k].reshape((100, 100)),
                      extent=(xmin, xmax, ymin, ymax), origin='lower')
            idx = (target == k)
            if idx.any():
                ax.scatter(x[idx], y[idx], marker='o', c='k')

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('auto')
        return axes

    def _plot_decision(self):
        target = self.data.target
        n_classes = len(target.unique())
        fig, axes = self._subplots(n_classes)

        x, y = self._get_xy_values()

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        # ToDo: Adjust edges
        xx, yy = self._get_meshgrid(xmin, xmax, ymin, ymax)
        Xfull = np.c_[xx.ravel(), yy.ravel()]
        Xfull = self._maybe_inverse(Xfull)
        decision = self.estimator.predict(Xfull)

        flatten = axes.flatten()

        for k, ax in zip(range(n_classes), flatten):
            ax.imshow(decision.reshape((100, 100)),
                      extent=(xmin, xmax, ymin, ymax), origin='lower')
            idx = (target == k)
            if idx.any():
                ax.scatter(x[idx], y[idx], marker='o', c='k')

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('auto')
        return axes


class RegressorPlotter(_BasePlotter):

    def plot(self):
        if len(self.data.data.columns) == 1:
            return self._plot_2d()
        elif len(self.data.data.columns) == 2:
            return self._plot_3d()
        else:
            raise ValueError

    def _plot_2d(self):
        x = self._get_x_values()

        # ToDo: 3D regression plot when ncols is 2 D
        fig, ax = self._subplots(1)

        xx = np.linspace(x.min(), x.max(), 100)
        ax.scatter(x, self.data.target, color='black')

        pred = self.estimator.predict(xx.reshape(len(xx), 1))
        ax.plot(xx, pred, color='blue')
        return ax

    def _plot_3d(self):
        x, y = self._get_xy_values()

        from mpl_toolkits.mplot3d import Axes3D     # noqa
        fig = self.plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, self.data.target, color='black')

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        xx, yy = self._get_meshgrid(xmin, xmax, ymin, ymax, step=10)
        Xfull = np.c_[xx.ravel(), yy.ravel()]
        pred = self.estimator.predict(Xfull).reshape(10, 10)
        ax.plot_wireframe(xx, yy, pred, color='blue')

        return ax
