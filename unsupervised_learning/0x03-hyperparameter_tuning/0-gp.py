#!/usr/bin/env python3
'''
Modulus that represents a noiseless 1D
Gaussian process
'''
import numpy as np


class GaussianProcess:
    '''
    Class that represent Gaussian process
    '''
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        '''
        Initialization of the class
        '''
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        '''
        Function that calculates the cov kernel mat
        '''
        cuadra = np.sum(X1 ** 2, 1).reshape(-1, 1) +\
            np.sum(X2 ** 2, 1) - 2 * np.matmul(X1, X2.T)
        expo = - (1 / (2 * (self.l ** 2))) * cuadra
        kali = (self.sigma_f ** 2) * np.exp(expo)
        return kali
