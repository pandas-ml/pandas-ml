#!/usr/bin/env python

import importlib

import numpy as np
import pandas as pd


class AccessorMethods(object):
    _module_name = None

    def __init__(self, df):
        self._df = df

        if self._module_name is None:
            return

        self._module = importlib.import_module(self._module_name)

        try:
            mobjs = self._module.__all__
        except AttributeError:
            return

        for mobj in mobjs:
            try:
                if not hasattr(self, mobj):
                    try:
                        setattr(self, mobj, getattr(self._module, mobj))
                    except AttributeError:
                        pass
            except NotImplementedError:
                pass

    @property
    def data(self):
        return self._df.data

    @property
    def target(self):
        return self._df.target

    @property
    def predicted(self):
        return self._df.predicted


def _attach_methods(cls, wrap_func, methods):
    try:
        module = importlib.import_module(cls._module_name)

        for method in methods:
            _f = getattr(module, method)
            setattr(cls, method, wrap_func(_f))

    except ImportError:
        pass
