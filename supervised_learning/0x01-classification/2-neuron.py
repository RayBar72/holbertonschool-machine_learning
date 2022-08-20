#!/usr/bin/env python3
'''Module that has class Neuron'''
import numpy as np


class Neuron():
    '''
    Class Neuron that defines a single
    neuron performing binary classification
    '''
    def __init__(self, nx):
        '''Class constructor'''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''Getter function of W'''
        return self.__W

    @property
    def b(self):
        '''Getter function of b'''
        return self.__b

    @property
    def A(self):
        '''Getter function of A'''
        return self.__A

    def forward_prop(self, X):
        '''
        Public method that calculates the forward
        propagation of the neuron
        '''
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
