#!/usr/bin/env python3
'''
Class that represents a gated recurrent unit
'''
import numpy as np


class GRUCell:
    '''
    Class that represents a gated recurrent unit
    '''
    def __init__(self, i, h, o):
        '''
        Initialization of gated recurrent unit
        '''
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    # Creating activation functions to no retyping
    def sigmond(self, x):
        '''
        Function sigmoid
        '''
        return 1 / (1 + np.exp(-x))

    def soft(self, x):
        '''
        Function softmax
        '''
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    # Forward propagation
    def forward(self, h_prev, x_t):
        '''
        Function performs forward propagation for one time step
        '''
        h = np.concatenate((h_prev.T, x_t.T), axis=0)

        zt = self.sigmond((h.T @ self.Wz) + self.bz)

        rt = self.sigmond((h.T @ self.Wr) + self.br)

        h = np.concatenate(((rt * h_prev).T, x_t.T), axis=0)

        ht = np.tanh((h.T @ self.Wh) + self.bh)

        h_next = (1 - zt) * h_prev + zt * ht

        y = self.soft((h_next @ self.Wy) + self.by)

        return h_next, y
