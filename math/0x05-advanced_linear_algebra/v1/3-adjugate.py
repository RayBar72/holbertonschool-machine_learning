#!/usr/bin/env python3
'''Modulus that calculates adjugate matrix'''
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    '''Function that calculates the cofactor matrix'''
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    if all([type(i) is list for i in matrix]) is False:
        raise TypeError('matrix must be a list of lists')

    if (matrix[0] and len(matrix) != len(matrix[0])) or matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == len(matrix[0]) == 1:
        return [[1]]

    mat = cofactor(matrix)

    largo = len(mat)

    retorno = []

    for i in range(largo):
        retornor = []
        for j in range(largo):
            retornor.append(mat[j][i])
        retorno.append(retornor)
    return retorno
