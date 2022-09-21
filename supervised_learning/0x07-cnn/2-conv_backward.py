#!/usr/bin/env python3
'''
Modulus that has a function that performs bkwr prop over a CNN
'''
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''
    Function that performs backward propagation over a convolutional
    layer of a neural network
    '''

    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = (0, 0)
    else:
        ph = int((sh * (h_prev - 1) - h_prev + kh) / 2)
        pw = int((sw * (w_prev - 1) - w_prev + kw) / 2)

    A_prev = np.pad(A_prev,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for img in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    filter = W[:, :, :, f]
                    dz = dZ[img, h, w, f]
                    sl_A = A_prev[img,
                                  h * sh:h * sh + kh,
                                  w * sw:w * sw + kw,
                                  :]
                    dW[:, :, :, f] += sl_A * dz
                    dA_prev[img,
                            h * sh:h * sh + kh,
                            w * sw:w * sw + kw,
                            :] += dz * filter
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    return dA_prev, dW, db
