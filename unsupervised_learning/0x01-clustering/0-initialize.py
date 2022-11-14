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
    if len(X.shape) != 2 or not isinstance(X, np.ndarray):
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    min = np.min(X, axis= 0)
    max = np.max(X, axis=0)

    init = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))

    return init
