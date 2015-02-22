#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.dummy as dummy

import expandas as expd
import expandas.util.testing as tm


class TestDummy(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.dummy.DummyClassifier, dummy.DummyClassifier)
        self.assertIs(df.dummy.DummyRegressor, dummy.DummyRegressor)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
