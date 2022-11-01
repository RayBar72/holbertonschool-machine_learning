#!/usr/bin/env python3
'''Modulus that calculates minor matrix'''
determinant = __import__('0-determinant').determinant


def minor(matrix):
    '''Function that calculates the minor matrix'''
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if all([type(i) is list for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")

    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == len(matrix[0]) == 1:
        return [[1]]

    mat = matrix

    largo = len(mat)

    dete = []
    for i in range(largo):
        deter = []
        for j in range(largo):
            mati = []
            for k in range(largo):
                matir = []
                for lo in range(largo):
                    if k != i and lo != j:
                        matir.append(mat[k][lo])
                if k != i:
                    mati.append(matir)
                # print(mati)
            deter.append(determinant(mati))
        dete.append(deter)
    return dete
