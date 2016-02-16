#!/usr/bin/env python

import numpy as np
import sklearn.datasets as datasets
import sklearn.learning_curve as lc
import sklearn.naive_bayes as nb
import sklearn.svm as svm

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestLearningCurve(tm.TestCase):

    def test_learning_curve(self):
        digits = datasets.load_digits()
        df = pdml.ModelFrame(digits)

        result = df.learning_curve.learning_curve(df.naive_bayes.GaussianNB())
        expected = lc.learning_curve(nb.GaussianNB(), digits.data, digits.target)

        self.assertEqual(len(result), 3)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])
        self.assert_numpy_array_almost_equal(result[2], expected[2])

    def test_validation_curve(self):
        digits = datasets.load_digits()
        df = pdml.ModelFrame(digits)

        param_range = np.logspace(-2, -1, 2)

        svc = df.svm.SVC(random_state=self.random_state)
        result = df.learning_curve.validation_curve(svc, 'gamma',
                                                    param_range)
        expected = lc.validation_curve(svm.SVC(random_state=self.random_state),
                                       digits.data, digits.target,
                                       'gamma', param_range)

        self.assertEqual(len(result), 2)
        self.assert_numpy_array_almost_equal(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
