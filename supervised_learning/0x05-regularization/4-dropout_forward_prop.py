#!/usr/bin/env python3
'''
Modulus that conducts forward propagation usig dropout
'''
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''
    Function  that conducts forward propagation using Dropout

    Parameters
    ----------
    X : TYPE numpy.ndarray
        DESCRIPTION. X is a numpy.ndarray of shape (nx, m) containing
        the input data for the network. nx - number of input features,
        m - number of data points
    weights : TYPE dictionary
        DESCRIPTION. Dictionary of the weights and biases of NN
    L : TYPE int
        DESCRIPTION. Number of layers
    keep_prob : TYPE float
        DESCRIPTION. probability that the node will be kept

    Returns
    -------
    A dictionary containing the outputs of each layer and the dropout
    mask used on each layer (see example for format).

    '''
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        w = weights['W' + str(i)]
        b = weights['b' + str(i)]
        a0 = cache['A' + str(i - 1)]
        Z = np.matmul(w, a0) + b
        if i == L:
            num = np.exp(Z)
            A = num / np.sum(num, axis=0, keepdims=True)
            cache['A' + str(i)] = A
        else:
            A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A *= D
            A /= keep_prob
            cache['A' + str(i)] = A
            cache['D' + str(i)] = D
    return cache
