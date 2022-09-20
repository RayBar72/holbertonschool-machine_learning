#!/usr/bin/env python3
'''
Modulus that has a function that performs fwrd prop over a CNN
'''
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''
    Function that performs forward propagation over a convolutional
    layer of a neural network
    '''

    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    c_new = b.shape[3]
    sh, sw = stride

    if padding == 'valid':
        ph, pw = (0, 0)
    else:
        ph = int((sh * (h_prev - 1) - h_prev + kh) / 2)
        pw = int((sw * (w_prev - 1) - w_prev + kw) / 2)

    ch = int(((h_prev + 2 * ph - kh) / sh) + 1)
    cw = int(((w_prev + 2 * pw - kw) / sw) + 1)
    conv_dim =(m, ch, cw, c_new)

    padded_img = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    conv = np.zeros(conv_dim)

    for i in range(conv_dim[1]):
        for j in range(conv_dim[2]):
            for n in range(conv_dim[3]):
                image_slice = padded_img[:,
                                         i * sh:i * sh + kh,
                                         j * sw:j * sw + kw]
                kernel = W[:, :, :, n]
                conv[:, i, j, n] = np.sum(image_slice * kernel, axis=(1, 2, 3))
    reto = activation(conv + b)
    return reto
