#!/usr/bin/env python3
'''
Module that defines a deep neural network
performing binary classification
'''
import numpy as np


class DeepNeuralNetwork():
    '''Class that defines a deep neural network'''
    def __init__(self, nx, layers):
        '''Class constructur'''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.nx = nx
        self.L = len(layers)
        self.cache = {}
        weights = {}
        for x in range(len(layers)):
            if layers[x] < 1:
                raise TypeError('layers must be a list of positive integers')
            if not isinstance(layers[x], int):
                raise TypeError('layers must be a list of positive integers')
            weights['b' + str(x + 1)] = np.zeros([layers[x], 1])
            if x == 0:
                He_et0 = np.random.randn(layers[x], nx) * np.sqrt(2 / nx)
                weights['W' + str(x + 1)] = He_et0
            if x > 0:
                He_et_p1 = np.random.rand(layers[x], layers[x - 1])
                He_et_p2 = np.sqrt(2 / layers[x - 1])
                weights['W' + str(x + 1)] = He_et_p1 * He_et_p2
        self.weights = weights
