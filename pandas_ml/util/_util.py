#!/usr/bin/env python


def _is_1d_varray(arr):
    """
    Check whether the array has single dimension like (x, ) or (x, 1)
    """
    return len(arr.shape) < 2 or arr.shape[1] == 1


def _is_1d_harray(arr):
    """
    Check whether the array has single dimension like (x, ) or (1, x)
    """
    return len(arr.shape) < 2 or arr.shape[0] == 1
