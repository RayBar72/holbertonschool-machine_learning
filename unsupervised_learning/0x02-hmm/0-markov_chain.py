#!/usr/bin/env python3
'''
Modulus that determines the probability
of a markov chain being in a particular
state after a specified number of iterations
'''
import numpy as np


def markov_chain(P, s, t=1):
    '''
    Modulus that determines the probability
    of a markov chain being in a particular
    state after a specified number of iterations
    '''
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    n, m = P.shape

    if n != m:
        return None

    if type(s) is not np.ndarray or s.shape != (1, n):
        return None

    if type(t) is not int or t < 0:
        return None

    reto = np.matmul(s, np.linalg.matrix_power(P, t))

    return reto
