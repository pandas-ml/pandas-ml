#!/usr/bin/env python

import datetime
import warnings

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.svm as svm

import expandas as expd
import expandas.util.testing as tm


class TestModelFrame(tm.TestCase):

    def test_version(self):
        self.assertTrue(len(expd.__version__) > 1)

    def test_frame_instance(self):

        df = expd.ModelFrame(datasets.load_digits())
        self.assertTrue(isinstance(df, expd.ModelFrame))

        train_df, test_df = df.cross_validation.train_test_split()

        self.assertTrue(isinstance(train_df, expd.ModelFrame))
        self.assertTrue(isinstance(test_df, expd.ModelFrame))
        self.assertTrue(isinstance(train_df.iloc[:, 2:3], expd.ModelFrame))

    def test_frame_init_df_series(self):
        # initialization by dataframe and no-named series
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                           index=['a', 'b', 'c'],
                           columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

        mdf = expd.ModelFrame(df, target=s)
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, df)
        self.assert_series_equal(mdf.target, s)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        s = pd.Series([1, 2, 3])
        with self.assertRaisesRegexp(ValueError, 'data and target must have equal index'):
            mdf = expd.ModelFrame(df, target=s)

        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='XXX')
        mdf = expd.ModelFrame(df, target=s)
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['XXX', 'A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, df)
        self.assert_series_equal(mdf.target, s)
        self.assertEqual(mdf.target_name, 'XXX')

    def test_frame_init_df_str(self):
        # initialization by dataframe and str
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                           index=['a', 'b', 'c'],
                           columns=['A', 'B', 'C'])

        mdf = expd.ModelFrame(df, target='A')
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 3))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, df[['B', 'C']])
        self.assert_series_equal(mdf.target, df['A'])
        self.assertEqual(mdf.target.name, 'A')
        self.assertEqual(mdf.target_name, 'A')

        msg = "Specified target 'X' is not included in data"
        with self.assertRaisesRegexp(ValueError, msg):
            mdf = expd.ModelFrame(df, target='X')

    def test_frame_init_dict_list(self):
        # initialization by dataframe and list
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                           index=['a', 'b', 'c'],
                           columns=['A', 'B', 'C'])
        s = [1, 2, 3]
        mdf = expd.ModelFrame(df, target=s)
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, df)
        expected = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        self.assert_series_equal(mdf.target, expected)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        df = {'A': [1, 2, 3],
              'B': [4, 5, 6],
              'C': [7, 8, 9]}
        s = [1, 2, 3]
        mdf = expd.ModelFrame(df, target=s)
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index([0, 1, 2]))
        self.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        expected = pd.DataFrame(df)
        self.assert_frame_equal(mdf.data, expected)
        expected = pd.Series([1, 2, 3], index=[0, 1, 2])
        self.assert_series_equal(mdf.target, expected)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        mdf = expd.ModelFrame(df, target='A')
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 3))
        self.assert_index_equal(mdf.index, pd.Index([0, 1, 2]))
        self.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        expected = pd.DataFrame(df)
        self.assert_frame_equal(mdf.data, expected[['B', 'C']])
        self.assert_series_equal(mdf.target, expected['A'])
        self.assertEqual(mdf.target.name, 'A')
        self.assertEqual(mdf.target_name, 'A')

        mdf = expd.ModelFrame({'A': [1, 2, 3],
                               'B': [4, 5, 6],
                               'C': [7, 8, 9]},
                              index=['a', 'b', 'c'],
                              columns=['A', 'B', 'C'])
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 3))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, mdf)
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_init_df_array_series(self):
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        mdf = expd.ModelFrame(np.array([[1, 2, 3], [4, 5, 6],
                                        [7, 8, 9]]), target=s,
                              index=['a', 'b', 'c'], columns=['A', 'B', 'C'])

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))

        expected = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6],
                                          [7, 8, 9]]),
                                index=['a', 'b', 'c'], columns=['A', 'B', 'C'])
        self.assert_frame_equal(mdf.data, expected)
        self.assert_series_equal(mdf.target, s)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_init_dict_list_series_index(self):
        # initialization by dataframe and list
        df = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        target = pd.Series([9, 8, 7], name='X', index=['a', 'b', 'c'])
        mdf = expd.ModelFrame(df, target=target)

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['X', 'A', 'B', 'C']))
        expected = pd.DataFrame(df, index=['a', 'b', 'c'])
        self.assert_frame_equal(mdf.data, expected)
        self.assert_series_equal(mdf.target, target)
        self.assertEqual(mdf.target_name, 'X')

    def test_frame_init_df_none(self):
        # initialization by dataframe and none
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                           index=['a', 'b', 'c'],
                           columns=['A', 'B', 'C'])

        mdf = expd.ModelFrame(df, target=None)
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 3))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, df)
        self.assertTrue(mdf.has_data())
        self.assertTrue(mdf.target is None)
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_data_none(self):
        msg = "ModelFrame must have either data or target"
        with self.assertRaisesRegexp(ValueError, msg):
            mdf = expd.ModelFrame(None)

        msg = "target must be list-like when data is None"
        with self.assertRaisesRegexp(ValueError, msg):
            mdf = expd.ModelFrame(None, target='X')

        # initialization without data
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        mdf = expd.ModelFrame(None, target=s)

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 1))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target']))
        self.assertTrue(mdf.data is None)
        self.assertFalse(mdf.has_data())
        self.assert_series_equal(mdf.target, s)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_slice(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        expected = ['.target'] + iris.feature_names
        self.assertEqual(df.columns.tolist(), expected)

        s = df['.target']
        self.assertTrue(isinstance(s, expd.ModelSeries))
        s = s[1:5]
        self.assertTrue(isinstance(s, expd.ModelSeries))

        s = df[['.target']]
        self.assertTrue(isinstance(s, expd.ModelFrame))

    def test_frame_data_proparty(self):
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                           index=['a', 'b', 'c'],
                           columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

        mdf = expd.ModelFrame(df, target=s)
        self.assertTrue(isinstance(mdf, expd.ModelFrame))

        new = pd.DataFrame({'X': [1, 2, 3],
                            'Y': [4, 5, 6]},
                            index=['a', 'b', 'c'],
                            columns=['X', 'Y'])
        # set data property
        mdf.data = new

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 3))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target', 'X', 'Y']))
        self.assert_frame_equal(mdf.data, new)
        self.assert_series_equal(mdf.target, s)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        new = expd.ModelFrame({'M': [1, 2, 3],
                               'N': [4, 5, 6]},
                              index=['a', 'b', 'c'],
                              columns=['M', 'N'])

        # set data property
        mdf.data = new

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 3))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target', 'M', 'N']))
        self.assert_frame_equal(mdf.data, new)
        self.assert_series_equal(mdf.target, s)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        new = pd.DataFrame({'.target': [1, 2, 3],
                            'K': [4, 5, 6]},
                            index=['a', 'b', 'c'])

        # unable to set data if passed value has the same column as the target
        msg = "Passed data has the same column name as the target '.target'"
        with self.assertRaisesRegexp(ValueError, msg):
            mdf.data = new

        # unable to set ModelFrame with target attribute
        msg = "Cannot update with ModelFrame which has target attribute"
        with self.assertRaisesRegexp(ValueError, msg):
            mdf.data = mdf

        # set delete property
        del mdf.data
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 1))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target']))
        self.assertTrue(mdf.data is None)
        self.assert_series_equal(mdf.target, s)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_target_proparty(self):
        df = pd.DataFrame({'A': [1, 2, 3],
                           'B': [4, 5, 6],
                           'C': [7, 8, 9]},
                           index=['a', 'b', 'c'],
                           columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        mdf = expd.ModelFrame(df, target=s)

        new = pd.Series([4, 5, 6], index=['a', 'b', 'c'], name='.target')
        # set target property
        mdf.target = new

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, df)
        self.assert_series_equal(mdf.target, new)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        with tm.assert_produces_warning(UserWarning):
            new = pd.Series([4, 5, 6], index=['a', 'b', 'c'], name='xxx')
            # set target property
            mdf.target = new

            self.assertTrue(isinstance(mdf, expd.ModelFrame))
            self.assertEqual(mdf.shape, (3, 4))
            self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
            self.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
            self.assert_frame_equal(mdf.data, df)
            self.assert_series_equal(mdf.target, new)
            self.assertEqual(mdf.target.name, '.target')
            self.assertEqual(mdf.target_name, '.target')

        new = pd.Series([4, 5, 6], name='.target')
        with self.assertRaisesRegexp(ValueError, 'data and target must have equal index'):
            mdf.target = new

        # set target property
        mdf.target = [7, 8, 9]

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['.target', 'A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, df)
        expected = pd.Series([7, 8, 9], index=['a', 'b', 'c'])
        self.assert_series_equal(mdf.target, expected)
        self.assertEqual(mdf.target.name, '.target')
        self.assertEqual(mdf.target_name, '.target')

        with self.assertRaisesRegexp(ValueError, 'Wrong number of items passed 2, placement implies 3'):
            mdf.target = [1, 2]

        # set target property
        mdf.target = None

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 3))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        self.assert_index_equal(mdf.columns, pd.Index(['A', 'B', 'C']))
        self.assert_frame_equal(mdf.data, df)
        self.assertEqual(mdf.target_name, '.target')

    def test_frame_delete_proparty(self):
        mdf = expd.ModelFrame(None, target=[1, 2, 3])
        msg = 'ModelFrame must have either data or target'

        with self.assertRaisesRegexp(ValueError, msg):
            del mdf.target

        with self.assertRaisesRegexp(ValueError, msg):
            mdf.target = None

        mdf = expd.ModelFrame([1, 2, 3])
        msg = 'ModelFrame must have either data or target'

        with self.assertRaisesRegexp(ValueError, msg):
            del mdf.data

        with self.assertRaisesRegexp(ValueError, msg):
            mdf.data = None

    def test_frame_target_object(self):
        df = pd.DataFrame({datetime.datetime(2014, 1, 1): [1, 2, 3],
                           datetime.datetime(2015, 1, 1): [4, 5, 6],
                           datetime.datetime(2016, 1, 1): [7, 8, 9]},
                           index=['a', 'b', 'c'])
        mdf = expd.ModelFrame(df, target=datetime.datetime(2016, 1, 1))

        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 3))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        expected = pd.DatetimeIndex(['2014-01-01', '2015-01-01', '2016-01-01'])
        self.assert_index_equal(mdf.columns, expected)
        self.assert_frame_equal(mdf.data, df.iloc[:, :2])
        expected = pd.Series([7, 8, 9], index=['a', 'b', 'c'])
        self.assert_series_equal(mdf.target, expected)
        self.assertEqual(mdf.target.name, datetime.datetime(2016, 1, 1))
        self.assertEqual(mdf.target_name, datetime.datetime(2016, 1, 1))

    def test_frame_target_object_set(self):

        df = pd.DataFrame({datetime.datetime(2014, 1, 1): [1, 2, 3],
                           datetime.datetime(2015, 1, 1): [4, 5, 6],
                           datetime.datetime(2016, 1, 1): [7, 8, 9]},
                           index=['a', 'b', 'c'])
        mdf = expd.ModelFrame(df)

        mdf.target = pd.Series(['A', 'B', 'C'], index=['a', 'b', 'c'], name=5)
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        expected = pd.Index([5, datetime.datetime(2014, 1, 1),
                             datetime.datetime(2015, 1, 1), datetime.datetime(2016, 1, 1)])
        self.assert_index_equal(mdf.columns, expected)
        self.assert_frame_equal(mdf.data, df)
        expected = pd.Series(['A', 'B', 'C'], index=['a', 'b', 'c'])
        self.assert_series_equal(mdf.target, expected)
        self.assertEqual(mdf.target.name, 5)
        self.assertEqual(mdf.target_name, 5)

        # name will be ignored if ModelFrame already has a target
        mdf.target = pd.Series([10, 11, 12], index=['a', 'b', 'c'], name='X')
        self.assertTrue(isinstance(mdf, expd.ModelFrame))
        self.assertEqual(mdf.shape, (3, 4))
        self.assert_index_equal(mdf.index, pd.Index(['a', 'b', 'c']))
        expected = pd.Index([5, datetime.datetime(2014, 1, 1),
                             datetime.datetime(2015, 1, 1), datetime.datetime(2016, 1, 1)])
        self.assert_index_equal(mdf.columns, expected)
        self.assert_frame_equal(mdf.data, df)
        expected = pd.Series([10, 11, 12], index=['a', 'b', 'c'])
        self.assert_series_equal(mdf.target, expected)
        self.assertEqual(mdf.target.name, 5)
        self.assertEqual(mdf.target_name, 5)

    def test_predict_proba(self):
        iris = datasets.load_iris()
        df = expd.ModelFrame(iris)

        models = ['SVC']
        for model in models:
            mod1 = getattr(df.svm, model)(probability=True, random_state=self.random_state)
            mod2 = getattr(svm, model)(probability=True, random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            result = df.predict(mod1)
            expected = mod2.predict(iris.data)

            self.assertTrue(isinstance(result, expd.ModelSeries))
            self.assert_numpy_array_almost_equal(result.values, expected)

            result = df.predict_proba(mod1)
            expected = mod2.predict_proba(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)
            self.assert_numpy_array_almost_equal(df.proba.values, expected)

            result = df.predict_log_proba(mod1)
            expected = mod2.predict_log_proba(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)
            self.assert_numpy_array_almost_equal(df.log_proba.values, expected)

            result = df.decision_function(mod1)
            expected = mod2.decision_function(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)
            self.assert_numpy_array_almost_equal(df.decision.values, expected)

            # not reset if estimator is identical
            df.fit(mod1)
            self.assertFalse(df._predicted is None)
            self.assertFalse(df._proba is None)
            self.assertFalse(df._log_proba is None)
            self.assertFalse(df._decision is None)

            # reset estimator
            mod3 = getattr(df.svm, model)(probability=True, random_state=self.random_state)
            df.fit(mod3)
            self.assertTrue(df._predicted is None)
            self.assertTrue(df._proba is None)
            self.assertTrue(df._log_proba is None)
            self.assertTrue(df._decision is None)

    def test_predict_automatic(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)

            iris = datasets.load_iris()
            df = expd.ModelFrame(iris)

            model = 'SVC'

            df = expd.ModelFrame(iris)
            mod1 = getattr(df.svm, model)(probability=True, random_state=self.random_state)
            mod2 = getattr(svm, model)(probability=True, random_state=self.random_state)

            df.fit(mod1)
            mod2.fit(iris.data, iris.target)

            # test automatically calls related methods
            with tm.assert_produces_warning(UserWarning):
                result = df.predicted
            expected = mod2.predict(iris.data)

            self.assertTrue(isinstance(result, expd.ModelSeries))
            self.assert_numpy_array_almost_equal(result.values, expected)

            # with tm.assert_produces_warning(UserWarning):
            result = df.proba
            expected = mod2.predict_proba(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

            with tm.assert_produces_warning(UserWarning):
                result = df.log_proba
            expected = mod2.predict_log_proba(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

            # with tm.assert_produces_warning(UserWarning):
            result = df.decision
            expected = mod2.decision_function(iris.data)

            self.assertTrue(isinstance(result, expd.ModelFrame))
            self.assert_index_equal(result.index, df.index)
            self.assert_numpy_array_almost_equal(result.values, expected)

        warnings.simplefilter("default")


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
