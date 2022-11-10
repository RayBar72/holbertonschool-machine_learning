#!/usr/bin/env python3
'''Modulus that calculates prior in continus function'''
import numpy as np
from scipy import special


def posterior(x, n, p1, p2):
    '''Function that calculates prior in continus function'''
    if not isinstance(n, int) or (n <= 0):
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or (x < 0):
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError('p1 must be a float in the range [0, 1]')
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError('p2 must be a float in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')

    P = (x - (p1 + p2)) / (n - (p1 + p2))
    like_0 = np.math.factorial(n) / (np.math.factorial(x) *
                                     np.math.factorial(n - x))
    like = like_0 * (P ** x) * ((1 - P) ** (n - x))
    Pr = special.beta(1, 1)
    inter = like * Pr
    margot = np.sum(inter)
    return inter / margot
