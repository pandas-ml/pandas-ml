#!/usr/bin/env python

import pandas as pd
import statsmodels.api as sm

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestStatsModelsDatasets(tm.TestCase):

    load_method = 'load'

    def test_anes96(self):
        data = getattr(sm.datasets.anes96, self.load_method)()
        df = pdml.ModelFrame(data)

        self.assertEqual(df.shape, (944, 6))
        self.assertEqual(df.target_name, 'PID')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_cancer(self):
        data = getattr(sm.datasets.cancer, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (301, 2))
        self.assertEqual(df.target_name, 'cancer')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_ccard(self):
        data = getattr(sm.datasets.ccard, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (72, 5))
        self.assertEqual(df.target_name, 'AVGEXP')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_committee(self):
        data = getattr(sm.datasets.committee, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (20, 6))
        self.assertEqual(df.target_name, 'BILLS104')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_copper(self):
        data = getattr(sm.datasets.copper, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (25, 6))
        self.assertEqual(df.target_name, 'WORLDCONSUMPTION')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_cpunish(self):
        data = getattr(sm.datasets.cpunish, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (17, 7))
        self.assertEqual(df.target_name, 'EXECUTIONS')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_elnino(self):
        data = getattr(sm.datasets.elnino, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)

    def test_engel(self):
        data = getattr(sm.datasets.engel, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (235, 2))
        self.assertEqual(df.target_name, 'income')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_grunfeld(self):
        data = getattr(sm.datasets.grunfeld, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (220, 5))
        self.assertEqual(df.target_name, 'invest')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_longley(self):
        data = getattr(sm.datasets.longley, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (16, 7))
        self.assertEqual(df.target_name, 'TOTEMP')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_macrodata(self):
        data = getattr(sm.datasets.macrodata, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)

    def test_modechoice(self):
        data = getattr(sm.datasets.modechoice, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (840, 7))
        self.assertEqual(df.target_name, 'choice')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_nile(self):
        data = getattr(sm.datasets.nile, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)

    def test_randhie(self):
        data = getattr(sm.datasets.randhie, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (20190, 10))
        self.assertEqual(df.target_name, 'mdvis')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_scotland(self):
        data = getattr(sm.datasets.scotland, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (32, 8))
        self.assertEqual(df.target_name, 'YES')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_spector(self):
        data = getattr(sm.datasets.spector, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (32, 4))
        self.assertEqual(df.target_name, 'GRADE')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_stackloss(self):
        data = getattr(sm.datasets.stackloss, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (21, 4))
        self.assertEqual(df.target_name, 'STACKLOSS')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_star98(self):
        data = getattr(sm.datasets.star98, self.load_method)()
        msg = 'Data must be 1-dimensional'
        with self.assertRaisesRegexp(Exception, msg):
            pdml.ModelFrame(data)

    def test_strikes(self):
        data = getattr(sm.datasets.strikes, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (62, 2))
        self.assertEqual(df.target_name, 'duration')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_sunspots(self):
        data = getattr(sm.datasets.sunspots, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)

    def test_fair(self):
        data = getattr(sm.datasets.fair, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (6366, 9))
        self.assertEqual(df.target_name, 'affairs')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_heart(self):
        data = getattr(sm.datasets.heart, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (69, 3))
        self.assertEqual(df.target_name, 'survival')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_statecrime(self):
        data = getattr(sm.datasets.statecrime, self.load_method)()
        df = pdml.ModelFrame(data)
        self.assertEqual(df.shape, (51, 5))
        self.assertEqual(df.target_name, 'murder')
        self.assert_index_equal(df.data.columns, pd.Index(data.exog_name))

    def test_co2(self):
        data = getattr(sm.datasets.co2, self.load_method)()
        msg = "Unable to read statsmodels Dataset without exog"
        with self.assertRaisesRegexp(ValueError, msg):
            pdml.ModelFrame(data)


class TestStatsModelsDatasets_LoadPandas(TestStatsModelsDatasets):

    load_method = 'load_pandas'

    def test_star98(self):
        data = sm.datasets.star98.load_pandas()
        msg = 'cannot copy sequence with size 2 to array axis with dimension 303'
        with self.assertRaisesRegexp(Exception, msg):
            pdml.ModelFrame(data)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
