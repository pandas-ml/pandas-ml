#!/usr/bin/env python

import pandas.compat as compat
from pandas.util.decorators import Appender, cache_readonly

import expandas.misc as misc


_shared_docs = dict()

_shared_docs['skaccessor'] = """
    Property to access ``sklearn.%(module)s``. See :mod:`expandas.skaccessors.%(module)s`
    """
_shared_docs['skaccessor_nolink'] = """
    Property to access ``sklearn.%(module)s``
    """

class AbstractModel(object):

    def _check_attr(self, estimator, method_name):
        if not hasattr(estimator, method_name):
            msg = "class {0} doesn't have {1} method"
            raise ValueError(msg.format(type(estimator), method_name))
        return getattr(estimator, method_name)

    def _call(self, estimator, method_name, *args, **kwargs):
        # must be overrided
        raise NotImplementedError

    _shared_docs['estimator_methods'] = """
        Call estimator's %(funcname)s method.

        Parameters
        ----------
        args : arguments passed to %(funcname)s method
        kwargs : keyword arguments passed to %(funcname)s method

        Returns
        -------
        %(returned)s
        """

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit', returned='returned : None or fitted estimator'))
    def fit(self, estimator, *args, **kwargs):
        return self._call(estimator, 'fit', *args, **kwargs)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='transform', returned='returned : transformed result'))
    def transform(self, estimator, *args, **kwargs):
        if isinstance(estimator, compat.string_types):
            return misc.transform_with_patsy(estimator, self, *args, **kwargs)
        transformed = self._call(estimator, 'transform', *args, **kwargs)
        return self._wrap_transform(transformed)

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='fit_transform', returned='returned : transformed result'))
    def fit_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'fit_transform', *args, **kwargs)
        return self._wrap_transform(transformed)

    def _wrap_transform(self, transformed):
        raise NotImplementedError

    @Appender(_shared_docs['estimator_methods'] %
              dict(funcname='inverse_transform', returned='returned : transformed result'))
    def inverse_transform(self, estimator, *args, **kwargs):
        transformed = self._call(estimator, 'inverse_transform', *args, **kwargs)
        return self._wrap_transform(transformed)

