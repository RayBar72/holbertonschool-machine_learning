#!/usr/bin/env python3
'''Function that adds two matrix'''
import numpy as np


def add_matrices2D(mat1, mat2):
    '''Adds two matrices to the end'''
    ma1 = np.array(mat1)
    ma2 = np.array(mat2)
    try:
        ma3 = np.add(ma1, ma2)
    except Exception as e:
        return None
    largo = list(ma3.shape)
    list0 = []
    for r in range(largo[0]):
        list1 = []
        for x in range(largo[1]):
            list1.append(ma3[r, x])
        list0.append(list1)
    return list0
