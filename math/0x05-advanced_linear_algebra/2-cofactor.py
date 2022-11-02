#!/usr/bin/env python3
'''Modulus that calculates cofactor matrix'''
minor = __import__('1-minor').minor


def cofactor(matrix):
    '''Function that calculates the cofactor matrix'''
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    if all([type(i) is list for i in matrix]) is False:
        raise TypeError('matrix must be a list of lists')

    if (matrix[0] and len(matrix) != len(matrix[0])) or matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    if all(len(matrix) == len(colum) for colum in matrix) is False:
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == len(matrix[0]) == 1:
        return [[1]]

    mat = minor(matrix)

    largo = len(mat)

    for i in range(largo):
        for j in range(largo):
            if i % 2 == 0:
                x = 1
            else:
                x = -1
            if j % 2 == 0:
                y = 1
            else:
                y = -1
            mat[i][j] = x * y * mat[i][j]
    return mat
