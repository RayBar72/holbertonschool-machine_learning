#!/usr/bin/env python3
'''
Modulus that has a function that performs forwar propagation over a
pooling layer of a NN
'''
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
    Function that performs forward propagation over a pooling layer of
    a neural network
    '''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    poolh = int((h_prev - kh) / sh) + 1
    poolw = int((w_prev - kw) / sw) + 1

    pool_dim = (m, poolh, poolw, c_prev)
    pooled = np.zeros(pool_dim)

    for i in range(pool_dim[1]):
        for j in range(pool_dim[2]):
            image_slice = A_prev[:,
                                 i * sh:i * sh + kh,
                                 j * sw:j * sw + kw]
            if mode == 'max':
                pooled[:, i, j] = np.max(image_slice, axis=1).max(axis=1)
            elif mode == 'avg':
                pooled[:, i, j] = np.mean(image_slice, axis=1).mean(axis=1)
    return pooled
