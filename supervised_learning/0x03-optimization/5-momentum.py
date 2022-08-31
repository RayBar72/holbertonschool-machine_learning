#!/usr/bin/env python3
'''Function that returns update_variables_momentum'''
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''Update variables with momentum'''
    vd = beta1 * v + (1 - beta1) * grad
    w = var - alpha * vd
    return w, vd
