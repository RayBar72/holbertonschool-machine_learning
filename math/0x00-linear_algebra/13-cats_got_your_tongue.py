#!/usr/bin/env python3
'''Function that concatenates two matrix'''
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''Concatenates two matrices'''
    mat3 = np.concatenate((mat1, mat2), axis=axis)
    return mat3
