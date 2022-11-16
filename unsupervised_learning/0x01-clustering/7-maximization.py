#!/usr/bin/env python3
'''
Modulus that calculates the maximization step
in the EM algorithm for GMM
'''
import numpy as np


def maximization(X, g):
    '''
    Function that calculates the maximization step
    in the EM algorithm for GMM
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None

    k, n = g.shape
    d = X.shape[1]

    x = np.sum(g, axis=0)
    x = np.sum(x)

    if x != n:
        return None, None, None

    sum_g = np.sum(g, axis=1)
    pi = (1 / n) * sum_g
    m = np.zeros(shape=(k, d))
    S = np.zeros(shape=(k, d, d))

    for i in range(k):
        m_i = np.matmul(g[i], X) / sum_g[i]
        m[i] = m_i
        X_m = X - m_i
        s_m = np.matmul(g[i] * X_m.T, X_m) / sum_g[i]
        S[i] = s_m

    return pi, m, S
