#!/usr/bin/env python3
'''Function that shuffle data X, Y'''
import numpy as np


def shuffle_data(X, Y):
    '''Shuffle data X, Y'''
    m = X.shape[0]
    shuf_vect = np.random.permutation(m)
    print(shuf_vect)
    x = X[shuf_vect]
    y = Y[shuf_vect]
    return x, y
