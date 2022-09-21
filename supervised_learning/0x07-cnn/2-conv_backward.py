#!/usr/bin/env python3
'''
Modulus that has a functuion that excutes a Back prop for a CNN
'''
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''
    Function that performs back propagation over a convolutional layer
    of a neural network
    '''
    