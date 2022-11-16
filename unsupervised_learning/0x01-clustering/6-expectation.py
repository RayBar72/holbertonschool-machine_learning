#!/usr/bin/env python3
'''
Modulus that calculates the expectation step in the EM algorithm for a GMM
'''
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    '''
    Function that calculates the expectation step in the
    EM algorithm for a GMM
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None

    if np.isclose(np.sum(pi), [1]) != 1:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if type(m) is not np.ndarray or m.shape != (k, d):
        return None, None

    if type(S) is not np.ndarray or S.shape != (k, d, d):
        return None, None

    Ns = np.zeros(shape=(k, n))

    for i in range(k):
        N = pdf(X, m[i], S[i])
        Ns[i] = N * pi[i]
    g = Ns / np.sum(Ns, axis=0)

    lo = np.sum(np.log(np.sum(Ns, axis=0)), axis=0)

    return g, lo
