#!/usr/bin/env python3
'''Modulus that calculates determinant matrix'''


def smaller_matrix(original_matrix, row, column):
    '''Function that makes a smaller matrix'''
    new_matrix = []
    for i in range(len(original_matrix)):
        if i == row:
            continue
        row_mati = []
        for j in range(len(original_matrix)):
            if j == column:
                continue
            row_mati.append(original_matrix[i][j])
        new_matrix.append(row_mati)
    return new_matrix


def determinant(matrix):
    '''Function that caculates the determianant of a matrix'''
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    if all([type(i) is list for i in matrix]) is False:
        raise TypeError('matrix must be a list of lists')

    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a square matrix')

    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    num_rows = len(matrix)

    if len(matrix) == 2:
        simple_determinant = matrix[0][0] * matrix[1][1] \
            - matrix[0][1] * matrix[1][0]
        return simple_determinant

    else:
        answer = 0
        num_columns = num_rows
        for j in range(num_columns):
            cofactor = (-1) ** (0 + j) * matrix[0][j] \
                * determinant(smaller_matrix(matrix, 0, j))
            answer += cofactor
        return answer
