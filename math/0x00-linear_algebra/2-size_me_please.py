#!/usr/bin/env python3
'''Function tha calculates the shape of a matrix'''
import numpy as np


def matrix_shape(matrix):
    '''Calculates the shape of a matrix.
    Parameters
    ----------
    matrix : array-like
        The matrix to calculate the shape of
    Returns
    -------
    int
        The shape of the of matrix'''
    m = np.array(matrix)
    r = list(m.shape)
    return r
