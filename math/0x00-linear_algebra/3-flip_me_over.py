#!/usr/bin/env python3
'''Function that traspose a matrix'''


def matrix_transpose(matrix):
    '''Transpose a matrix such that it can be applied to another matrix.
    :param matrix: The matrix to transpose
    :return: The transposed matrix
    '''
    row = len(matrix)
    col = len(matrix[0])
    matrir = []
    for i in range(col):
        list0 = []
        for j in range(row):
            list0.append(matrix[j][i])
        matrir.append(list0)
    return matrir
