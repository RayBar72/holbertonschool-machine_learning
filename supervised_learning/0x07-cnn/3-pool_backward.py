#!/usr/bin/env python3
'''
Modulus that perform back prop over a pooling layer in a NN
'''
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
    Function that performs back propagation over a pooling layer of a
    neural network
    '''
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for img in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    y_s = h * sh
                    y_e = h * sh + kh
                    x_s = w * sw
                    x_e = w * sw + kw
                    da = dA[img, h, w, c]
                    if mode == 'max':
                        a_slice = A_prev[img, y_s:y_e, x_s:x_e, c]
                        max_s = (a_slice == np.max(a_slice))
                        dA_prev[img,
                                y_s:y_e,
                                x_s:x_e,
                                c] += np.multiply(max_s, da)
                    elif mode == 'avg':
                        average = da / (kh * kw)
                        x = np.ones(kernel_shape.shape) * average
                        dA_prev[img,
                                y_s:y_e,
                                x_s:x_e,
                                c] += x
    return dA_prev
