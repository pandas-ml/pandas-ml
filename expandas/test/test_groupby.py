#!/usr/bin/env python

import datetime
import warnings

import pandas as pd

import expandas as expd
import expandas.util.testing as tm


class TestModelFrameGroupBy(tm.TestCase):

    def test_frame_groupby(self):
        df = pd.DataFrame({'A': [1, 2, 1, 2],
                           'B': [4, 5, 6, 7],
                           'C': [7, 8, 9, 10]},
                           columns=['A', 'B', 'C'])
        s = pd.Series([1, 2, 3, 4])

        mdf = expd.ModelFrame(df, target=s)
        self.assertTrue(isinstance(mdf, expd.ModelFrame))

        grouped = mdf.groupby('A')
        self.assertTrue(isinstance(grouped, expd.core.groupby.ModelFrameGroupBy))

        df = grouped.get_group(1)
        self.assertTrue(isinstance(df, expd.ModelFrame))

        expected = pd.Series([1, 3], index=[0, 2], name='.target')
        self.assert_series_equal(df.target, expected)
        self.assertTrue(isinstance(df.target, expd.ModelSeries))


class TestModelSeriesGroupBy(tm.TestCase):

    def test_series_groupby(self):
        s = expd.ModelSeries([1, 2, 1, 2], name='X')
        self.assertTrue(isinstance(s, expd.ModelSeries))

        grouped = s.groupby([1, 1, 1, 2])
        self.assertTrue(isinstance(grouped, expd.core.groupby.ModelSeriesGroupBy))

        gs = grouped.get_group(1)
        self.assertTrue(isinstance(gs, expd.ModelSeries))
        expected = pd.Series([1, 2, 1], index=[0, 1, 2], name='X')
        self.assert_series_equal(gs, expected)
        self.assertEqual(gs.name, 'X')



if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
