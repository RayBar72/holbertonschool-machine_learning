#!/usr/bin/env python3
"""
Modulus that makes FP for a Bi RNN
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Funcson that makew FP
    """
    t, m, i = X.shape
    steps = range(t)
    _, h = h_0.shape

    H_f = np.zeros((t + 1, m, h))
    H_b = np.zeros((t + 1, m, h))

    H_f[0] = h_0
    H_b[t] = h_t

    for s in steps:
        H_f[s+1] = bi_cell.forward(H_f[s], X[s])

    for r in range(t-1, -1, -1):
        H_b[r] = bi_cell.backward(H_b[r+1], X[r])
    H = np.concatenate((H_f[1:], H_b[:t]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
