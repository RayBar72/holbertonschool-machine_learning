#!/usr/bin/env python3
'''
Modulus that calculates the total intra-cluster
variance for a data set
'''
import numpy as np


def variance(X, C):
    '''
    Function that calculates the total intra-cluster
    variance for a data set
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    distancia = np.linalg.norm(X - C[:, np.newaxis], axis=-1)
    min_distancia = np.min(distancia, axis=0)
    var = np.sum(min_distancia ** 2)

    return var
