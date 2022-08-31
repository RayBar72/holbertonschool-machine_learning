#!/usr/bin/env python3
'''Function that update_variables_Adam'''


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''Function that update_variables_Adam'''
    vd = beta1 * v + (1 - beta1) * grad
    sd = beta2 * s + (1 - beta2) * (grad ** 2)
    vc = vd / (1 - beta1 ** t)
    sc = sd / (1 - beta2 ** t)
    var = var - alpha * (vc / ((sc ** (1 / 2)) + epsilon))
    return var, vd, sd
