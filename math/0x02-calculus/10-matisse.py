#!/usr/bin/env python3
'''Function that casculates polynomial derivatives'''


def poly_derivative(poly):
    '''Function that caculates polynomial derivatives'''
    aux = []
    if (type(poly) is not list) or (len(poly) == 0):
        return None
    if (not all(isinstance(n, int) or isinstance(n, float) for n in poly)):
        return None
    for i in range(1, len(poly)):
        aux.append(i * poly[i])
    if all(a == 0 for a in aux):
        return [0]
    return aux