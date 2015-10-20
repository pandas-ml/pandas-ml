#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestSklearnBase(tm.TestCase):

    def test_load_dataset(self):
        iris = datasets.load_iris()
        df = pdml.ModelFrame(iris)
        self.assertEqual(df.shape, (150, 5))

        msg = "'target' can't be specified for sklearn.datasets"
        with self.assertRaisesRegexp(ValueError, msg):
            mdf = pdml.ModelFrame(iris, target='X')


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
