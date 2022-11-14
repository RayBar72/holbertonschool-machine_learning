#!/usr/bin/env python3
'''
Modulus that performs K-means on a dataset
'''
import numpy as np


def kmeans(X, k, iterations=1000):
    '''
    Function that performs K-means on a dataset
    '''
    if type(k) is not int or k < 1:
        return None, None

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(iterations) is not int or iterations < 1:
        return None, None

    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    C = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))

    for i in range(iterations):
        C_c = np.copy(C)
        distancias = np.linalg.norm(X - C[:, np.newaxis], axis=-1)
        clss = distancias.argmin(axis=0)

        for j in range(k):
            if X[clss == j].size == 0:
                C[j] = np.random.uniform(min, max, size=(1, X.shape[1]))
            else:
                C[j] = X[clss == j].mean(axis=0)

        distancias = np.linalg.norm(X - C[:, np.newaxis], axis=-1)
        clss = distancias.argmin(axis=0)

        if (C == C_c).all():
            break

    return C, clss
