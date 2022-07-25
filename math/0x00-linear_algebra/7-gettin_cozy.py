#!/usr/bin/env python3
'''Function that concatenates two matrix'''
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    '''Function that concatenates two matrices'''
    try:
        mat3 = np.concatenate((mat1, mat2), axis=axis)
    except Exception as e:
        return None
    lista0 = []
    largo = list(mat3.shape)
    for r in range(largo[0]):
        lista1 = []
        for x in range(largo[1]):
            lista1.append(mat3[r, x])
        lista0.append(lista1)
    return lista0
