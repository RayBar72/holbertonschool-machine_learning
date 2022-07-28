#!/usr/bin/env python3
'''Function that slice_like_a_ninja'''


def np_slice(matrix, axes={}):
    '''Function that slice'''
    mat_tup_sl = [slice(None, None, None)] * matrix.ndim
    for k, v in sorted(axes.items()):
        sv = slice(*v)
        mat_tup_sl[k] = sv
    matrix = matrix[tuple(mat_tup_sl)]
    return matrix
