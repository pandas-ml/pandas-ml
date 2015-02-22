#!/usr/bin/env python

import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets

import expandas as expd
import expandas.util.testing as tm


class TestModelFrame(tm.TestCase):

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

        msg = "ModelFrame doesn't have target '.target'"
        with self.assertRaisesRegexp(ValueError, msg):
            mdf.target
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
        with self.assertRaisesRegexp(AttributeError, "can't delete attribute"):
            del mdf.data


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


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
