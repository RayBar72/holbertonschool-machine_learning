#!/usr/bin/env python3
'''
Modulus that  updates the weights of a neural network with
Dropout regularization using gradient descent
'''
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''
    Function  that updates the weights of a neural network with
    Dropout regularization using gradient descent

    Parameters
    ----------
    Y : TYPE numpy.ndarray
        DESCRIPTION. Y is a one-hot numpy.ndarray of shape (classes, m)
        that contains the correct labels for the data
    weights : TYPE dictionary
        DESCRIPTION. Dictionary of the weights and biases of the neural network
    cache : TYPE dictionary
        DESCRIPTION. Dictionary of the outputs and dropout masks of
        each layer of the neural network
    alpha : TYPE float
        DESCRIPTION. Learning rate
    keep_prob : TYPE float
        DESCRIPTION. Probabily that a node will be kept
    L : TYPE int
        DESCRIPTION. Number of layers of DNN

    Returns
    -------
    None.

    '''
    m = Y.shape[1]
    for i in reversed(range(1, L + 1)):
        w = weights['W' + str(i)]
        b = weights['b' + str(i)]
        a0 = cache['A' + str(i - 1)]
        a = cache['A' + str(i)]
        if i == L:
            dz = a - Y
            W = w
        else:
            d = cache['D' + str(i)]
            da = 1 - (a * a)
            dz = np.matmul(W.T, dz)
            dz = dz * da * d
            dz = dz / keep_prob
            W = w
        dw = np.matmul(a0, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(i)] = w - alpha * dw.T
        weights['b' + str(i)] = b - alpha * db
