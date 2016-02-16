#!/usr/bin/env python

import importlib

import pandas as pd


def _get_version(package):
    try:
        importlib.import_module(package)
        return 'Installed'
    except ImportError:
        return 'Not installed'


def info():
    packages = ['sklearn', 'statsmodels', 'seaborn', 'xgboost']
    versions = [_get_version(p) for p in packages]
    df = pd.DataFrame({'Installed': versions}, index=packages)
    return df
