#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import statsmodels.api as sm

import expandas as expd
import expandas.util.testing as tm


class TestStatsModelsDatasets(tm.TestCase):

    def test_anes96(self):
        data = sm.datasets.anes96.load()
        df = expd.ModelFrame(data)

        self.assertEqual(df.shape, (944, 6))
        self.assertEqual(df.target_name, 'PID')

    def test_cancer(self):
        data = sm.datasets.cancer.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (301, 2))
        self.assertEqual(df.target_name, 'cancer')

    def test_ccard(self):
        data = sm.datasets.ccard.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (72, 5))
        self.assertEqual(df.target_name, 'AVGEXP')

    def test_committee(self):
        data = sm.datasets.committee.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (20, 6))
        self.assertEqual(df.target_name, 'BILLS104')

    def test_copper(self):
        data = sm.datasets.copper.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (25, 6))
        self.assertEqual(df.target_name, 'WORLDCONSUMPTION')

    def test_cpunish(self):
        data = sm.datasets.cpunish.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (17, 7))
        self.assertEqual(df.target_name, 'EXECUTIONS')

    def test_elnino(self):
        data = sm.datasets.elnino.load()
        return

    def test_engel(self):
        data = sm.datasets.engel.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (235, 2))
        self.assertEqual(df.target_name, 'income')

    def test_grunfeld(self):
        data = sm.datasets.grunfeld.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (220, 5))
        self.assertEqual(df.target_name, 'invest')

    def test_longley(self):
        data = sm.datasets.longley.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (16, 7))
        self.assertEqual(df.target_name, 'TOTEMP')

    def test_macrodata(self):
        data = sm.datasets.macrodata.load()
        return

    def test_modechoice(self):
        data = sm.datasets.modechoice.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (840, 7))
        self.assertEqual(df.target_name, 'choice')

    def test_nile(self):
        data = sm.datasets.nile.load()
        return

    def test_randhie(self):
        data = sm.datasets.randhie.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (20190, 10))
        self.assertEqual(df.target_name, 'mdvis')

    def test_scotland(self):
        data = sm.datasets.scotland.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (32, 8))
        self.assertEqual(df.target_name, 'YES')

    def test_spector(self):
        data = sm.datasets.spector.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (32, 4))
        self.assertEqual(df.target_name, 'GRADE')

    def test_stackloss(self):
        data = sm.datasets.stackloss.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (21, 4))
        self.assertEqual(df.target_name, 'STACKLOSS')

    def test_star98(self):
        data = sm.datasets.star98.load()
        return

    def test_strikes(self):
        data = sm.datasets.strikes.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (62, 2))
        self.assertEqual(df.target_name, 'duration')

    def test_sunspots(self):
        data = sm.datasets.sunspots.load()
        return

    def test_fair(self):
        data = sm.datasets.fair.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (6366, 9))
        self.assertEqual(df.target_name, 'affairs')

    def test_heart(self):
        data = sm.datasets.heart.load()
        return

    def test_statecrime(self):
        data = sm.datasets.statecrime.load()
        df = expd.ModelFrame(data)
        self.assertEqual(df.shape, (51, 5))
        self.assertEqual(df.target_name, 'murder')

    def test_co2(self):
        data = sm.datasets.co2.load()
        return


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
