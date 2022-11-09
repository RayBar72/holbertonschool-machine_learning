#!/usr/bin/env python3
'''Modulus that calculates PCA'''
import numpy as np


def pca(X, var=0.95):
    '''
    Function that calculates weights matrix
    '''
    U, sigma, V = np.linalg.svd(X)
    Var = np.cumsum(sigma) / np.sum(sigma)
    r = (np.argwhere(Var >= var))[0][0]
    W = V[:r + 1].T
    return W
