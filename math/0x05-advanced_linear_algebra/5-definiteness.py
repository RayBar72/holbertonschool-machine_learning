#!/usr/bin/env python3
'''Modulus that calculates definiteness of a matrix'''
import numpy as np


def definiteness(matrix):
    '''Function that calculates definiteness of a matrix'''
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) < 2:
        return None

    h, w = matrix.shape
    if h != w:
        return None

    if np.all(np.linalg.eigvals(matrix) > 0):
        return 'Positive definite'
    elif np.all(np.linalg.eigvals(matrix) >= 0):
        return 'Positive semi-definite'
    elif np.all(np.linalg.eigvals(matrix) < 0):
        return 'Negative definite'
    elif np.all(np.linalg.eigvals(matrix) <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
