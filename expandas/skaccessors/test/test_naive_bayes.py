#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import sklearn.datasets as datasets
import sklearn.naive_bayes as nb

import expandas as expd
import expandas.util.testing as tm


class TestNaiveBayes(tm.TestCase):

    def test_objectmapper(self):
        df = expd.ModelFrame([])
        self.assertIs(df.naive_bayes.GaussianNB, nb.GaussianNB)
        self.assertIs(df.naive_bayes.MultinomialNB, nb.MultinomialNB)
        self.assertIs(df.naive_bayes.BernoulliNB, nb.BernoulliNB)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
