#!/usr/bin/env python

import numpy as np
import pandas as pd

from expandas.core.accessor import AccessorMethods, _attach_methods


class PipelineMethods(AccessorMethods):
    """
    Accessor to ``sklearn.pipeline``.
    """

    _module_name = 'sklearn.pipeline'

    @property
    def make_pipeline(self):
        # not included in __all__
        return self._module.make_pipeline

    @property
    def make_union(self):
        # not included in __all__
        return self._module.make_union