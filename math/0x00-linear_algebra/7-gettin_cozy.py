#!/usr/bin/env python3
'''Function that concatenates two matrix'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''Function that concatenates two matrices'''
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            matR1 = [x[:] for x in mat1]
            matR2 = [x[:] for x in mat2]
            matR = matR1 + matR2
            return matR
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            matC = [mat1[x] + mat2[x] for x in range(len(mat1))]
            return matC
    else:
        return None
