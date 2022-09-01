#!/usr/bin/env python3
'''
Function that normalizas an unactivated output of a NN
'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''
    Function that normalizas an unactivated output of a NN

    Parameters
    ----------
    Z : numpy.ndarray
        Array of (m,n) that should be normalized.
    gamma : numpy.ndarray
        scales used for batch normalization.
    beta : numpy.ndarray
        offsets used for batch normalization.
    epsilon : Float
        small number to avoid zero division.

    Returns
    -------
    Z_n : numpy.darray
        Normalized matrix.

    '''
    mu = np.mean(Z, axis=0)
    sigma_2 = np.var(Z, axis=0)
    norma = (Z - mu) / np.sqrt(sigma_2 + epsilon)
    Z_n = gamma * norma + beta
    return Z_n
