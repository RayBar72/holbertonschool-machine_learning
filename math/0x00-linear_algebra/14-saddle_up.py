#!/usr/bin/env python3
'''Function that multiplies two matrices'''
import numpy as np


def np_matmul(mat1, mat2):
    '''Multiplies two matrices'''
    mat3 = np.dot(mat1, mat2)
    return mat3
