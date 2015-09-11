#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.qda as qda

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestQDA(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.qda.QDA, qda.QDA)

    def test_QDA(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])

        df = pdml.ModelFrame(X, target=y)

        mod1 = df.qda.QDA()
        mod2 = qda.QDA()

        df.fit(mod1)
        mod2.fit(X, y)

        result = df.predict(mod1)
        expected = mod2.predict(X)

        self.assertTrue(isinstance(result, pdml.ModelSeries))
        self.assert_numpy_array_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
