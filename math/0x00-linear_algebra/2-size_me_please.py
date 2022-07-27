#!/usr/bin/env python3
'''Function tha calculates the shape of a matrix'''


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
    if type(matrix[0]) != list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])
