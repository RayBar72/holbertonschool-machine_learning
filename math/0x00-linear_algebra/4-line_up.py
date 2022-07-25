#!/usr/bin/env python3
'''Function that add two array'''
import numpy as np


def add_arrays(arr1, arr2):
    '''Add two arrays'''
    ar1 = np.array(arr1)
    ar2 = np.array(arr2)
    try:
        ar3 = np.add(arr1, arr2)
    except Exception as e:
        return None
    return list(ar3)
