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
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for x in range(self.L):
            if layers[x] < 1 or type(layers[x]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["b" + str(x + 1)] = np.zeros((layers[x], 1))
            if x == 0:
                He_et_0 = np.random.randn(layers[x], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(x + 1)] = He_et_0
            if x > 0:
                He_et_1 = np.random.randn(layers[x], layers[x - 1])
                He_et_2 = np.sqrt(2 / layers[x - 1])
                He_et_3 = He_et_1 * He_et_2
                self.__weights["W" + str(x + 1)] = He_et_3

    @property
    def L(self):
        '''L getter function'''
        return self.__L

    @property
    def weights(self):
        '''weight getter function'''
        return self.__weights

    @property
    def cache(self):
        '''cache getter function'''
        return self.__cache

    def forward_prop(self, X):
        '''
        Calculates the forward propagation
        of the neural network
        '''
        m = self.__L
        x = self.__cache['A0'] = X
        for i in range(m):
            w = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            z = np.matmul(w, x) + b
            a = 1 / (1 + np.exp(-z))
            self.__cache['A' + str(i + 1)] = a
            x = a
        return a, self.__cache

    def cost(self, Y, A):
        '''
        Calculates the cost of the model
        using logistic regression
        '''
        m = - (1 / Y.shape[1])
        x = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        C = m * x
        return C

    def evaluate(self, X, Y):
        '''
        Evaluates the neural network’s predictions
        '''
        a, _ = self.forward_prop(X)
        P = np.where(a >= 0.5, 1, 0)
        return P, self.cost(Y, a)
