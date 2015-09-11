#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.random_projection as rp

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestRandomProjection(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.random_projection.GaussianRandomProjection, rp.GaussianRandomProjection)
        self.assertIs(df.random_projection.SparseRandomProjection, rp.SparseRandomProjection)
        self.assertIs(df.random_projection.johnson_lindenstrauss_min_dim, rp.johnson_lindenstrauss_min_dim)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
