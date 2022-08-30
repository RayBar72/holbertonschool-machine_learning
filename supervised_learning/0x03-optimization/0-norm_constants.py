#!/usr/bin/env python3
'''
Modulus that calculates parameters for normalization process
'''
import numpy as np


def normalization_constants(X):
    '''
    Calculates the normalization constant values
    '''
    ave = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return ave, std
