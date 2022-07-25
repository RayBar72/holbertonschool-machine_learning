#!/usr/bin/env python3
'''
Function that performs addition, subtraction, multiplication and division
'''
import numpy as np


def np_elementwise(mat1, mat2):
    '''Perform the addition, subtraction, multiplication and division'''
    listo = [
        np.add(mat1, mat2),
        np.subtract(mat1, mat2),
        np.multiply(mat1, mat2),
        np.divide(mat1, mat2)
    ]
    topo = tuple(listo)
    return topo
