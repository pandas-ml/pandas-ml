#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.ensemble as ensemble

import expandas as expd
import expandas.util.testing as tm


class TestEnsemble(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.ensemble.AdaBoostClassifier, ensemble.AdaBoostClassifier)
        self.assertIs(df.ensemble.AdaBoostRegressor, ensemble.AdaBoostRegressor)
        self.assertIs(df.ensemble.BaggingClassifier, ensemble.BaggingClassifier)
        self.assertIs(df.ensemble.BaggingRegressor, ensemble.BaggingRegressor)
        self.assertIs(df.ensemble.ExtraTreesClassifier, ensemble.ExtraTreesClassifier)
        self.assertIs(df.ensemble.ExtraTreesRegressor, ensemble.ExtraTreesRegressor)
        self.assertIs(df.ensemble.GradientBoostingClassifier, ensemble.GradientBoostingClassifier)
        self.assertIs(df.ensemble.GradientBoostingRegressor, ensemble.GradientBoostingRegressor)
        self.assertIs(df.ensemble.RandomForestClassifier, ensemble.RandomForestClassifier)
        self.assertIs(df.ensemble.RandomTreesEmbedding, ensemble.RandomTreesEmbedding)
        self.assertIs(df.ensemble.RandomForestRegressor, ensemble.RandomForestRegressor)




if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
