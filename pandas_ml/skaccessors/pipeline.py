#!/usr/bin/env python

from pandas_ml.core.accessor import _AccessorMethods


class PipelineMethods(_AccessorMethods):
    """
    Accessor to ``sklearn.pipeline``.
    """

    _module_name = 'sklearn.pipeline'

    @property
    def make_pipeline(self):
        """``sklearn.pipeline.make_pipeline``"""
        # not included in __all__
        return self._module.make_pipeline

    @property
    def make_union(self):
        """``sklearn.pipeline.make_union``"""
        # not included in __all__
        return self._module.make_union
