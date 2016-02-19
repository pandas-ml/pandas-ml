#!/usr/bin/python
# -*- coding: utf8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from enum import Enum, IntEnum  # pip install enum34


class Backend(Enum):
    Matplotlib = 1
    Seaborn = 2


class Axis(IntEnum):
    Actual = 1
    Predicted = 0


BACKEND_DEFAULT = Backend.Matplotlib
SUM_NAME_DEFAULT = '__all__'
DISPLAY_SUM_DEFAULT = True
TRUE_NAME_DEFAULT = 'Actual'
PRED_NAME_DEFAULT = 'Predicted'
CLASSES_NAME_DEFAULT = 'Classes'
COLORBAR_TRIG = 10
