#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.lda as lda

import pandas_ml as pdml
import pandas_ml.util.testing as tm


class TestLDA(tm.TestCase):

    def test_objectmapper(self):
        df = pdml.ModelFrame([])
        self.assertIs(df.lda.LDA, lda.LDA)

    def test_LDA(self):
        diabetes = datasets.load_diabetes()
        df = pdml.ModelFrame(diabetes)

        models = ['LDA']
        for model in models:
            mod1 = getattr(df.lda, model)()
            mod2 = getattr(lda, model)()

            df.fit(mod1)
            mod2.fit(diabetes.data, diabetes.target)

            result = df.predict(mod1)
            expected = mod2.predict(diabetes.data)
            self.assertTrue(isinstance(result, pdml.ModelSeries))
            self.assert_numpy_array_almost_equal(result.values, expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
