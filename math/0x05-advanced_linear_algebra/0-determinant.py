#!/usr/bin/env python3
'''Modulus that calculates determinant matrix'''


def determinant(matrix):
    '''Function that caculates the determianant of a matrix'''
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if all([type(i) is list for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")

    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if matrix == [[]]:
        return 1
    mat = matrix

    largo = len(mat)
    diagonal = []
    abajo = [0]
    derecha = [0]

    for i in range(largo):
        for j in range(largo):
            if i == j:
                diagonal.append(mat[i][j])
                try:
                    abajo.append(mat[i + 1][j])
                except Exception as e:
                    pass
                try:
                    derecha.append(mat[i][j + 1])
                except Exception as e:
                    pass

    # print(diagonal)
    # print(abajo)
    # print(derecha)
    resultados = []
    for i in range(largo):
        try:
            multi = abajo[i] / diagonal[i - 1]
        except Exception as e:
            multi = 0
        x = multi * derecha[i]
        y = diagonal[i] - x
        resultados.append(y)
    # print(resultados)
    z = 1
    for j in resultados:
        z *= j
    return z
