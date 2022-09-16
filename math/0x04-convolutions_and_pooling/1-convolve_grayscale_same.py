#!/usr/bin/env python3
'''
Modulus that has a function that performs valid convolution grayscale images
'''
import numpy as np


def convolve_grayscale_same(images, kernel):
    '''
    Function that performs a valid convolution grayscale images:
        images: np.ndarray. images to be convoluted
        kernel. np.ndarray. filter to be used
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = int(np.ceil((kh - 1) / 2))
    pw = int(np.ceil((kh - 1) / 2))
    padded_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    conv_dim = (m, h, w)
    conv = np.zeros(conv_dim)

    for i in range(conv_dim[1]):
        for j in range(conv_dim[2]):
            image_slice = padded_img[:, i:i + kh, j:j + kw]
            conv[:, i, j] = np.tensordot(image_slice, kernel,
                                         axes=([1, 2], [0, 1]))
    return conv
