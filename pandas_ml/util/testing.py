#!/usr/bin/env python

import numpy as np
import pandas as pd
import pandas.compat as compat

import pandas.util.testing as tm
from pandas.util.testing import *

class TestCase(tm.TestCase):

    @property
    def random_state(self):
        return np.random.RandomState(1234)

    def assert_numpy_array_almost_equal(self, a, b):
        return np.testing.assert_array_almost_equal(a, b)
