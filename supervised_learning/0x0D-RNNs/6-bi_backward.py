#!/usr/bin/env python3
"""
Modulus that creates Bi-directional Cell of a RNN
"""

import numpy as np


class BidirectionalCell:
    """Class Bi-directional cell"""
    def __init__(self, i, h, o):
        """
        Initializer constructor of Bi-directional cell
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Funtion of forward propagation
        """
        h = np.concatenate((h_prev, x_t), axis=1)

        h = np.tanh(h @ self.Whf + self.bhf)

        return h

    def backward(self, h_next, x_t):
        """
        Function that makes BP
        """

        h = np.concatenate((h_next, x_t), axis=1)

        h = np.tanh((h @ self.Whb) + self.bhb)

        return h
