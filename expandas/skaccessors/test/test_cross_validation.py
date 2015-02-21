#!/usr/bin/env python

from __future__ import unicode_literals

import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.cross_validation as cv

import expandas as expd
import expandas.util.testing as tm


class TestCrossValidation(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.cross_validation.KFold, cv.KFold)
        self.assertIs(df.cross_validation.LeaveOneLabelOut, cv.LeaveOneLabelOut)
        self.assertIs(df.cross_validation.LeaveOneOut, cv.LeaveOneOut)
        self.assertIs(df.cross_validation.LeavePLabelOut, cv.LeavePLabelOut)
        self.assertIs(df.cross_validation.LeavePOut, cv.LeavePOut)
        self.assertIs(df.cross_validation.StratifiedKFold, cv.StratifiedKFold)
        self.assertIs(df.cross_validation.ShuffleSplit, cv.ShuffleSplit)
        self.assertIs(df.cross_validation.ShuffleSplit, cv.ShuffleSplit)

    def test_iterate(self):
        df = expd.ModelFrame(datasets.load_iris())
        kf = df.cross_validation.KFold(4, n_folds=2, random_state=self.random_state)
        for train_df, test_df in df.cross_validation.iterate(kf):
            self.assertTrue(isinstance(train_df, expd.ModelFrame))
            self.assertTrue(isinstance(test_df, expd.ModelFrame))
            self.assert_index_equal(df.columns, train_df.columns)
            self.assert_index_equal(df.columns, test_df.columns)

            self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])

    def test_train_test_split(self):

        df = expd.ModelFrame(datasets.load_digits())
        self.assertTrue(isinstance(df, expd.ModelFrame))

        train_df, test_df = df.cross_validation.train_test_split()
        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        self.assertTrue(df.shape[0], train_df.shape[0] + test_df.shape[1])
        self.assertTrue(df.shape[1], train_df.shape[1])
        self.assertTrue(df.shape[1], test_df.shape[1])

        self.assert_index_equal(df.columns, train_df.columns)
        self.assert_index_equal(df.columns, test_df.columns)

        # can't retain index
        # concatenated = pd.concat([train_df, test_df])
        # concatenated = concatenated.sort_index()
        # self.assert_frame_equal(df, concatenated)

    def test_cross_val_score(self):
        import sklearn.svm as svm
        digits = datasets.load_digits()

        df = expd.ModelFrame(digits)
        clf = svm.SVC(kernel=str('linear'), C=1)
        result = df.cross_validation.cross_val_score(clf, cv=5)
        expected = cv.cross_val_score(clf, digits.data, y=digits.target, cv=5)
        self.assert_numpy_array_almost_equal(result, expected)

    def test_permutation_test_score(self):
        import sklearn.svm as svm
        iris = datasets.load_iris()

        df = expd.ModelFrame(iris)
        clf = svm.SVC(kernel=str('linear'), C=1)
        result = df.cross_validation.permutation_test_score(clf, cv=5)
        expected = cv.permutation_test_score(clf, iris.data, y=iris.target, cv=5)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], expected[0])
        self.assert_numpy_array_almost_equal(result[1], expected[1])
        self.assertEqual(result[2], expected[2])

    def test_check_cv(self):
        iris = datasets.load_iris()

        df = expd.ModelFrame(iris)
        result = df.cross_validation.check_cv(cv=5)
        self.assertTrue(isinstance(result, cv.KFold))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
