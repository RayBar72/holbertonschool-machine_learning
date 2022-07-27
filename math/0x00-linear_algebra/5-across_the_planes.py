#!/usr/bin/env python3
'''Function that adds two matrix'''


def add_matrices2D(mat1, mat2):
    '''Adds two matrices to the end'''
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    mat0 = []
    for i in range(len(mat1)):
        list0 = []
        for j in range(len(mat1[0])):
            list0.append(mat1[i][j] + mat2[i][j])
        mat0.append(list0)
    return mat0
