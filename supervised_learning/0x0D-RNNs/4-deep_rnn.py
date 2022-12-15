#!/usr/bin/env python3
"""Modulus that makes FP to a deep RNN"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Function that makes foward propagation
    """
    Y = []
    t, m, i = X.shape
    _, _, h = h_0.shape
    time_step = range(t)
    layers = len(rnn_cells)

    H = np.zeros((t+1, layers, m, h))
    H[0, :, :, :] = h_0

    for s in time_step:
        for l in range(layers):
            if l == 0:
                h_next, y_pred = rnn_cells[l].forward(H[s, l], X[s])
            else:
                h_next, y_pred = rnn_cells[l].forward(H[s, l], h_next)
            H[s+1, l, :, :] = h_next
        Y.append(y_pred)
    Y = np.array(Y)
    return H, Y
