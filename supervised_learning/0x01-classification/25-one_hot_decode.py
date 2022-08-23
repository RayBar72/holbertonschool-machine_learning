#!/usr/bin/env python3
'''
Modulus that converts a one-hot matrix
into a vector of labels
'''
import numpy as np


def one_hot_decode(one_hot):
    '''
    Function that converts a one-hot
    matrix into a vector of labels
    '''
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    return np.argmax(one_hot, axis=0)
