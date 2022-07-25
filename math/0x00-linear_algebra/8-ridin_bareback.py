#!/usr/bin/env python3
'''Function tha multiplies two matrices'''
import numpy as np


def mat_mul(mat1, mat2):
    '''Multiplies two matrices'''
    try:
        mat3 = np.matmul(mat1, mat2)
    except Exception as e:
        return None
    largo = list(mat3.shape)
    lista0 = []
    for r in range(largo[0]):
        lista1 = []
        for x in range(largo[1]):
            lista1.append(mat3[r, x])
        lista0.append(lista1)
    return lista0
