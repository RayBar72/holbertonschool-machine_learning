#!/usr/bin/env python3
'''Function that calculates the integral'''


def poly_integral(poly, C=0):
    '''Function that calculates the integral of polynomial'''
    ls = []
    # Validate the list
    if type(poly) != list:
        return None
    if len(poly) == 0:
        return ls.append(C)
    if (not all(isinstance(x, (int, float)) for x in poly)):
        return None
    if not isinstance(C, (int, float)) or C is None:
        return None
    ls.append(C)
    if len(poly) == 0:
        return ls
    for i, x in enumerate(poly):
        ls.append(x / (i + 1))
    for i in range(len(ls)):
        if ls[i] % 1 == 0:
            ls[i] = int(ls[i])
    if all(x == 0 for x in ls):
        return [0]
    if len(ls) != 0:
        for i in range(len(ls) - 1, -1, - 1):
            if ls[i] == 0:
                ls.pop()
            else:
                break
    return ls
