#!/usr/bin/env python3
'''
Modulus that tests for the optimum number of clusters by variance
'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''
    Function that tests for the optimum
    number of clusters by variance
    '''
    try:
        if type(X) is not np.ndarray or len(X.shape) != 2:
            return None, None

        if type(kmin) is not int or kmin <= 0 or kmin >= X.shape[0]:
            return None, None

        if kmax is not None and (type(kmax) is not int or kmax <= 0):
            return None, None

        if kmax is not None and kmin >= kmax:
            return None, None

        if kmax is None:
            kmax = X.shape[0]

        if type(iterations) is not int or iterations <= 0:
            return None, None

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

    except Exception:
        return None, None
