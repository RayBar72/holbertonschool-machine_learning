#!/usr/bin/env python3
'''Modulus that calculates inverse matrix'''
adjugate = __import__('3-adjugate').adjugate
determinat = __import__('0-determinant').determinant


def inverse(matrix):
    '''Function that calculates the inverse matrix'''
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    if all([type(i) is list for i in matrix]) is False:
        raise TypeError('matrix must be a list of lists')

    if (matrix[0] and len(matrix) != len(matrix[0])) or matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    if all(len(matrix) == len(colum) for colum in matrix) is False:
        raise ValueError('matrix must be a non-empty square matrix')

    dete = determinat(matrix)

    if dete == 0:
        return None

    adjunta = adjugate(matrix)
    # print(adjunta)

    retorno = [[x / dete for x in row] for row in adjunta]

    return retorno
