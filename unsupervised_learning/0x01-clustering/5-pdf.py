#!/usr/bin/env python3
'''
Modulus that calculates the probability density function of
a Gaussian distribution
'''
import numpy as np


def pdf(X, m, S):
    '''
    Function that calculates the probability density function
    of a Gaussian distribution
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    n, d = X.shape

    if type(m) is not np.ndarray or len(m.shape) != 1 or m.shape[0] != d:
        return None

    if type(S) is not np.ndarray or S.shape != (d, d):
        return None

    det_S = np.linalg.det(S)
    S_I = np.linalg.inv(S)
    X_m = X - m

    exp_ = np.exp((- 1 / 2) * np.sum(X_m * np.matmul(X_m, S_I), axis=1))
    mul = 1 / (((2 * np.pi) ** (d / 2)) * (det_S ** (1 / 2)))
    res = mul * exp_

    res_aj = np.where(res < 1e-300, 1e-300, res)

    return res_aj
