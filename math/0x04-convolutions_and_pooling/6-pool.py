#!/usr/bin/env python3
'''
Modulus that has a function that performs valid convolution grayscale images
'''
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''
    Function that performs a pooling of images:
        images: np.ndarray. images to be convoluted
        kernel_shape. tuple with kener height and weight
        stride. tuple, steps at the filter is moving
        mode. str. max for maxpool, and avg for avgpool
    '''
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    poolh = int((h - kh) / sh) + 1
    poolw = int((w - kw) / sw) + 1

    pool_dim = (m, poolh, poolw, c)
    pooled = np.zeros(pool_dim)

    for i in range(pool_dim[1]):
        for j in range(pool_dim[2]):
            image_slice = images[:,
                                 i * sh:i * sh + kh,
                                 j * sw:j * sw + kw]
            if mode == 'max':
                pooled[:, i, j] = np.max(image_slice, axis=1).max(axis=1)
            elif mode == 'avg':
                pooled[:, i, j] = np.mean(image_slice, axis=1).mean(axis=1)
    return pooled
