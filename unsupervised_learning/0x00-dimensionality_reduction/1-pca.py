#!/usr/bin/env python3
'''Modulus that calculates PCA'''
import numpy as np


def pca(X, ndim):
    '''Function that calculates PCA'''
    x_m = X - np.mean(X, axis=0)
    U, sigma, V = np.linalg.svd(x_m)
    W = V[:ndim].T
    T = np.matmul(x_m, W)
    return T
