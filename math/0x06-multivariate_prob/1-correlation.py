#!/usr/bin/env python3
'''

Modulus that calculates correlation matrix
'''
import numpy as np


def correlation(C):
    '''
    Function that calculates correlation matix
    :param C: np.ndarray containing covariance matrix
    :return: correlation matrix
    '''

    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    diago = np.diag(np.absolute(C)) ** (1 / 2)
    # print(diago)
    n = C.shape[0]
    retorno = np.ndarray(shape=(n, n), dtype=np.float64)

    for i in range(C.shape[0]):
        for j in range(C.shape[0]):
            x = diago[i]
            y = diago[j]
            # print('{}, {}, {}'.format(C[i, j], x, y))
            z = C[i, j] / (x * y)
            retorno[i, j] = z
    return retorno
