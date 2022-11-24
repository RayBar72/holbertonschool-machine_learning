#!/usr/bin/env python3
'''
Modulus that determines the steady state
probabilities for a regular markov chain
'''
import numpy as np


def regular(P):
    '''
    Modulus that determines the steady state
    probabilities for a regular markov chain
    '''
    if type(P) is not np.ndarray or P.ndim != 2:
        return None

    n, m = P.shape

    if n != m:
        return None

    if np.any(np.sum(P, axis=1)) != 1:
        return None

    w, v = np.linalg.eig(P.T)

    one = np.argwhere(np.isclose(w, 1))
    if len(one) != 1:
        return None

    steady = v[:, one[0]] / np.sum(v[:, one[0]])

    if 0 is steady:
        return None

    return steady.T
