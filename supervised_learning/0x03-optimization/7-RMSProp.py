#!/usr/bin/env python3
'''Function that returns RMSProp'''
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''Function that makes the RMSProp update'''
    sd = beta2 * s + (1 - beta2) * (grad ** 2)
    rms = var - alpha * (grad / (sd ** (1 / 2) + epsilon))
    return rms, sd
