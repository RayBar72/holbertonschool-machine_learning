#!/usr/bin/env python3
'''
Modulus that initializes variables for a Gaussian Mixture Mode
'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    '''
    Function that initializes variables for a Gaussian Mixture Mode
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if type(k) is not int or k < 1:
        return None, None, None

    n, d = X.shape

    pi = np.ones(shape=(k,)) * (1 / k)

    m, _ = kmeans(X, k)

    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)

    return pi, m, S
