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
        self.L = len(layers)
        self.cache = {}
        for x in range(self.L):
            if layers[x] < 1 or not isinstance(layers[x], int):
                raise TypeError('layers must be a list of positive integers')
            