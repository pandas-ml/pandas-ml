#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.tree as tree

import expandas as expd
import expandas.util.testing as tm


class TestTree(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.tree.DecisionTreeClassifier, tree.DecisionTreeClassifier)
        self.assertIs(df.tree.DecisionTreeRegressor, tree.DecisionTreeRegressor)
        self.assertIs(df.tree.ExtraTreeClassifier, tree.ExtraTreeClassifier)
        self.assertIs(df.tree.ExtraTreeRegressor, tree.ExtraTreeRegressor)
        self.assertIs(df.tree.export_graphviz, tree.export_graphviz)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
