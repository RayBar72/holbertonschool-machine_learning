#!/usr/bin/env python3
'''Function that calculates the integral'''


def poly_integral(poly, C=0):
    '''Function that calculates the integral of polynomial'''
    if type(poly) != list:
        return None
    if type(C) is not int:
        return None
    ls = []
    largo = len(poly)
    if largo == 0:
        return ls.append(C)
    for i in range(largo):
        if i == 0:
            ls.append(C)
            ls.append(poly[i])
        else:
            ls.append(poly[i] / (i + 1))
    return ls
