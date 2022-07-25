#!/usr/bin/env python3
'''Function that traspose a matrix'''
import numpy as np


def matrix_transpose(matrix):
    '''Transpose a matrix such that it can be applied to another matrix.
    :param matrix: The matrix to transpose
    :return: The transposed matrix
    '''
    m = np.array(matrix).T
    s = list(m.shape)
    l = []
    for r in range(s[0]):
        l2 = []
        for x in range(s[1]):
            l2.append(m[r][x])
        l.append(l2)
    return l
