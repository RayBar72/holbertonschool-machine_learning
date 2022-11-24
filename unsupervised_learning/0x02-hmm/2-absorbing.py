#!/usr/bin/env python3
'''
Modulus that determines if a markov chain is absorbing
'''
import numpy as np


def absorbing(P):
    '''
    Function that determines if a markov chain is absorbing
    '''
    if type(P) is not np.ndarray or P.ndim != 2:
        return False

    n, m = P.shape

    if n != m:
        return False

    if np.any(np.sum(P, axis=1)) != 1:
        return False

    x = np.diag(P)
    if not np.any(x == 1):
        return False
    if np.all(x == 1):
        return True
    try:
        x = int(np.argwhere(x != 1)[0])
        In = P[:x, :x]
        On = P[:x, x:]
        R = P[x:, :x]
        Q = P[x:, x:]
        Id = np.identity(Q.shape[0])
        Qp = Id - Q
        F = np.linalg.inv(Qp)
        Ter = np.matmul(F, R)
        return True
    except Exception:
        return False
