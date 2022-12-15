#!/usr/bin/env python3
'''
Modulus that represents a simple RNN
'''
import numpy as np


def rnn(rnn_cell, X, h_0):
    '''
    Function that performs forward propagation for a simple RNN
    '''
    # Shapes - t, max steps - m, batch size - i, dims data - h, dim hidden
    t, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros(shape=(t + 1, m, h))
    Y = []

    H[0, :, :] = h_0

    for i in range(t):
        h_n, y_p = rnn_cell.forward(H[i], X[i])
        H[i + 1, :, :] = h_n
        Y.append(y_p)

    Y = np.array(Y)
    return H, Y
