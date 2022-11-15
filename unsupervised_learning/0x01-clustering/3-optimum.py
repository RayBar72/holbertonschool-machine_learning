#!/usr/bin/env python3
'''
that tests for the optimum number of clusters by variance
'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''
    Function that tests for the optimum
    number of clusters by variance
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(kmin) is not int or kmin < 1 or kmin >= X.shape[0]:
        return None, None

    if type(kmax) is not int or kmax < 1 or kmax >= X.shape[0]:
        return None, None

    if kmin >= kmax:
        return None, None

    if type(iterations) is not int or iterations < 1:
        return None, None

    try:
        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))
            var_d = variance(X, C)
            if k == kmin:
                var_k = var_d
            d_vars.append(var_k - var_d)

        return results, d_vars

    except Exception as e:
        return None, None
