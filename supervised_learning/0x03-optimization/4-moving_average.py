#!/usr/bin/env python3
'''Function that returns a list with moving_average'''
import numpy as np


def moving_average(data, beta):
    '''Function that returns a list with moving_average'''
    lista = []
    vt = 0
    for i, d in enumerate(data):
        vt = beta * vt + ((1 - beta) * d)
        bias = 1 - (beta ** (i + 1))
        vt_ad = vt / bias
        lista.append(vt_ad)
    return lista
