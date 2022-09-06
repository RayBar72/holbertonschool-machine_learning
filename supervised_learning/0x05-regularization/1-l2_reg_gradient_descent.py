#!/usr/bin/env python3
'''
Modulus that updates the weights and biases of a neural
network using gradient descent with L2 regularization
'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''
    Function that updates the weights and biases of a neural
    network using gradient descent with L2 regularization

    Parameters
    ----------
    Y : TYPE numpy.ndarray
        DESCRIPTION. Y is a one-hot numpy.ndarray of shape (classes, m)
        that contains the correct labels for the data
    weights : TYPE dictionary
        DESCRIPTION. Dictionary of the weights and biases of the
        neural network
    cache : TYPE dictionary
        DESCRIPTION. Dictionary of the outputs of each layer of
        the neural network
    alpha : TYPE float
        DESCRIPTION. Learning rate
    lambtha : TYPE float
        DESCRIPTION. L2 regularization parameter
    L : TYPE int
        DESCRIPTION. layers of the network

    Returns
    -------
    None.

    '''
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1 / m) * np.matmul(dz, cache['A' + str(i - 1)].T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dA = cache['A' + str(i - 1)] * (1 - cache['A' + str(i - 1)])
        dz = np.matmul(weights['W' + str(i)].T, dz) * dA
        weights['W' + str(i)] = weights[
            'W' + str(i)] * (1 - (alpha * lambtha) / m) - (alpha * dw)
        weights['b' + str(i)] = weights[
            'b' + str(i)] - (alpha * db)
