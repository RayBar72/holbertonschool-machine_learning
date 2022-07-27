#!/usr/bin/env python3
'''Function tha multiplies two matrices'''


def mat_mul(mat1, mat2):
    '''Multiplies two matrices'''
    row1 = len(mat1)
    col1 = len(mat1[0])
    row2 = len(mat2)
    col2 = len(mat2[0])
    if col1 != row2:
        return None
    mat0 = []
    for i in range(row1):
        list0 = []
        for j in range(col2):
            x = 0
            for k in range(row2):
                x += mat1[i][k] * mat2[k][j]
            list0.append(x)
        mat0.append(list0)
    return mat0
