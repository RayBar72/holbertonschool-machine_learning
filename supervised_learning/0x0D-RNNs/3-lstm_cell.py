#!/usr/bin/env python3
'''
Modulus that creates a LSTM
'''
import numpy as np


class LSTMCell:
    '''
    Class that represents LSTM
    '''
    def __init__(self, i, h, o):
        '''
        Function that init LSTM
        '''
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

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

    def forward(self, h_prev, c_prev, x_t):
        '''
        Function performs forward propagation for one time step
        '''
        h = np.concatenate((h_prev.T, x_t.T), axis=0)

        f = self.sigmond((h.T @ self.Wf) + self.bf)

        it = self.sigmond((h.T @ self.Wu) + self.bu)

        cct = np.tanh((h.T @ self.Wc) + self.bc)

        c_next = f * c_prev + it * cct

        ot = self.sigmond((h.T @ self.Wo) + self.bo)

        h_next = ot * np.tanh(c_next)

        y = self.soft((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
