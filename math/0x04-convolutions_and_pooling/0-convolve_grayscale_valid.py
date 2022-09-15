#!/usr/bin/env python3
'''
Modulus that has a function that performs valid convolution grayscale images
'''
import numpy as np


def convolve_grayscale_valid(images, kernel):
    '''
    Function that performs a valid convolution grayscale images:
        images: np.ndarray. images to be convoluted
        kernel. np.ndarray. filter to be used
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape

    conv_dim = (m, h - kh + 1, w - kw + 1)
    conv = np.zeros(conv_dim)

    for i in range(conv_dim[1]):
        for j in range(conv_dim[2]):
            image_slice = images[:, i:i + kh, j:j + kw]
            conv[:, i, j] = np.tensordot(image_slice, kernel,
                                         axes=([1, 2], [0, 1]))
    return conv
