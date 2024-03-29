#!/usr/bin/env python3
'''Function that casculates polynomial derivatives'''


def poly_derivative(poly):
    '''Function that caculates polynomial derivatives'''
    ls = []
    if type(poly) is not list:
        return None
    if len(poly) == 0:
        return None
    if (not all(isinstance(x, (int, float)) for x in poly)):
        return None
    for i in range(1, len(poly)):
        ls.append(poly[i] * i)
    if all(x == 0 for x in ls):
        return [0]
    return ls
