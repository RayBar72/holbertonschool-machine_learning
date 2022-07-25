#!/usr/bin/env python3
'''Function that concatenates two arrays'''
import numpy as np


def cat_arrays(arr1, arr2):
    '''Concatenate two arrays into a single array'''
    return np.concatenate((arr1, arr2), axis=0)
