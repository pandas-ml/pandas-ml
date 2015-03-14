#!/usr/bin/env python

import pandas as pd
from pandas.util.decorators import Appender, cache_readonly

from expandas.core.frame import ModelFrame
from expandas.core.series import ModelSeries


@Appender(pd.core.groupby.GroupBy.__doc__)
def groupby(obj, by, **kwds):
    if isinstance(obj, ModelSeries):
        klass = ModelSeriesGroupBy
    elif isinstance(obj, ModelFrame):
        klass = ModelFrameGroupBy
    else:  # pragma: no cover
        raise TypeError('invalid type: %s' % type(obj))

    return klass(obj, by, **kwds)


class ModelSeriesGroupBy(pd.core.groupby.SeriesGroupBy):
    pass

class ModelFrameGroupBy(pd.core.groupby.DataFrameGroupBy):
    pass
