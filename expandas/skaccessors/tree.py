#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas.util.decorators import cache_readonly

from expandas.core.accessor import AccessorMethods


class TreeMethods(AccessorMethods):
    _module_name = 'sklearn.tree'

