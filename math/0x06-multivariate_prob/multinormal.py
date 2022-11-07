#!/usr/bin/env python3
'''
Modulus that represents Multivariate Normal
Distribution
'''
import numpy as np


class MultiNormal():
    '''
    Class that represents multivariate normal
    distribution
    '''
    def __init__(self, data):
        '''
        Init function
        '''
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        d, n = data.shape

        self.mean = np.mean(data, axis=1).reshape(d, 1)

        X_m = data - self.mean

        self.cov = np.matmul(X_m, X_m.T) / (n - 1)

    def pdf(self, x):
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')

        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError('x must have the'
                             ' shape ({d}, 1)'.format(x.shape[0]))

        x_m = self.mean
        sigma = self.cov
        dim = x.shape[0]

        expo_1 = np.matmul(np.linalg.inv(sigma), x - x_m)
        expo_2 = (- 1 / 2) * np.matmul((x - x_m).T, expo_1)
        expo = np.sum(np.exp(expo_2))
        deto = np.sum(np.linalg.det(sigma) ** (- 1 / 2))
        pillo = np.sum((2 * np.pi) ** (- dim / 2))

        return pillo * deto * expo
