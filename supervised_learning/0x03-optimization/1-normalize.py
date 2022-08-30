#!/usr/bin/env python3
'''Function that normalize a matrix'''
import numpy as np


def normalize(X, m, s):
    '''Normalize a matrix'''
    x = np.subtract(X, m)
    ret = np.divide(x, s)
    return ret
