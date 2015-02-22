#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class SVMMethods(AccessorMethods):
    _module_name = 'sklearn.svm'

    def l1_min_c(self, *args, **kwargs):
        func = self._module.l1_min_c
        data = self.data
        target = self.target
        l1_min_c = func(data.values, y=target.values, *args, **kwargs)
        return l1_min_c

    @property
    def libsvm(self):
        raise NotImplementedError

    @property
    def liblinear(self):
        raise NotImplementedError

    @property
    def libsvm_sparse(self):
        raise NotImplementedError


