#!/usr/bin/env python3
'''Modulus that calculates likelihood'''
import numpy as np


def likelihood(x, n, P):
    '''Function that calculates likelihoo'''
    if n < 0 or type(n) is not int:
        raise ValueError('n must be a positive integer')

    if type(x) is not int or x < 0:
        raise ValueError('x must be an integer \
            that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')

    if type(P) is not np.ndarray:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')

    fact = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x))
    m_1 = P ** x
    m_2 = (1 - P) ** (n - x)

    return fact * m_1 * m_2
