#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class LDAMethods(AccessorMethods):
    _module_name = 'sklearn.lda'
