#!/usr/bin/env python3
'''
Module that converts a numeric label
vector into a one-hot matrix
'''
import numpy as np


def one_hot_encode(Y, classes):
    '''
    Function that converts a numeric label
    in to a one-hot matrix
    '''
    if len(Y) == 0:
        return None
    if not isinstance(Y, np.ndarray):
        return None
    if not isinstance(classes, int):
        return None
    if classes <= np.amax(Y):
        return None
    retorno = np.zeros([classes, len(Y)])
    retorno[Y, np.arange(len(Y))] = 1
    return retorno
