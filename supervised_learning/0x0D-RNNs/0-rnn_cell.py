#!/usr/bin/env python3
'''
Modulus that represents a simple RNN
'''
import numpy as np


class RNNCell():
    '''
    Function that represents a simple RNN
    '''
    def __init__(self, i, h, o):
        '''
        Initialization of the simple RNN
        '''
        # Dimensiones - i = data, h = hiden, o = outputs
        # Instancias públicas
        # El ejercicio prevee xw, no wx por tanto (i + h, h)
        # wy es (h, o)
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        # b son vectores de rango 1, columna ws
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        '''
        Function that performs simple RNN forward propagation
        '''
        x = (np.concatenate((h_prev, x_t), axis=1) @ self.Wh) + self.bh
        h_next = np.tanh(x)
        y = h_next @ self.Wy + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
