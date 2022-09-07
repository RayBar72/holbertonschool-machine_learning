#!/usr/bin/env python3
'''
Modulus that calculates the cost of a neural network with L2
regularization
'''
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''
    Function that calculates the cost of a neural network with
    L2 regularization

    Parameters
    ----------
    cost : TYPE Function
        DESCRIPTION. Costo of the network without L2 regularization
    lambtha : TYPE float
        DESCRIPTION. Regularization parameter
    weights : TYPE dictionary
        DESCRIPTION. Dictionary of the weights and biases
        (numpy.ndarrays) of the neural network
    L : TYPE int
        DESCRIPTION. Number of layers in neural network
    m : TYPE int
        DESCRIPTION. Number of data points

    Returns
    -------
    The costo of the network accounting L2 regularization.

    '''
    W = [weights['W' + str(i + 1)] for i in range(L)]
    W = [np.linalg.norm(w) ** 2 for w in W]
    cost = cost + (lambtha * sum(W) / (2 * m))
    return cost
