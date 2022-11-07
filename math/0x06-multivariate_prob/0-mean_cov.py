#!/usr/bin/env python3
'''
Modulus that calculates the mean and covariance of a data set
'''
import numpy as np


def mean_cov(X):
    '''
    Function that calculates the mean and covariance of a data set
    :param X: numpy.ndarray containing data set
    :return: mean and covarince
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')

    n, d = X.shape
    x = np.mean(X, axis=0).reshape(1, d)

    X_m = X - x

    Y = np.matmul(X_m.T, X_m) / (n - 1)

    return x, Y
