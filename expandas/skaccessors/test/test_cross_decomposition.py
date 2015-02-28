#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.cross_decomposition as cd

import expandas as expd
import expandas.util.testing as tm


class TestCrossDecomposition(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.cross_decomposition.PLSRegression, cd.PLSRegression)
        self.assertIs(df.cross_decomposition.PLSCanonical, cd.PLSCanonical)
        self.assertIs(df.cross_decomposition.CCA, cd.CCA)
        self.assertIs(df.cross_decomposition.PLSSVD, cd.PLSSVD)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
