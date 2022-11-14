#!/usr/bin/env python3
'''
Modulus that  initializes cluster centroids for K-means
'''
import numpy as np


def initialize(X, k):
    '''
    Function that initializes cluster centroids
    for K-means
    '''
    if type(k) is not int or k < 1:
        return None

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    init = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))

    return init
