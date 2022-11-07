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
        '''
        Function that calculates probability density
        of a multivarete normal
        :param x: Specific point
        :return: PDF value
        '''
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')

        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.
                             format(self.cov.shape[0]))
        if x.shape[1] != 1:
            raise ValueError('x must have the shape ({}, 1)'.
                             format(self.cov.shape[0]))
        if x.shape[0] != self.cov.shape[0]:
            raise ValueError('x must have the shape ({}, 1)'.
                             format(self.cov.shape[0]))

        x_m = self.mean
        sigma = self.cov

        d = x.shape[0]
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        X = x - x_m

        uno = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        dos = np.dot(-(X).T, inv)
        tres = np.dot(dos, X / 2)
        cua = np.exp(tres)
        pdf = float(uno * cua)

        return pdf
