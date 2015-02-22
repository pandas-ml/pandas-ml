#!/usr/bin/env python

import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.preprocessing as pp

import expandas as expd
import expandas.util.testing as tm


class TestPreprocessing(tm.TestCase):

    def test_objectmapper(self):
        pass


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
