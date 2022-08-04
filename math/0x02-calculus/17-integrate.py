#!/usr/bin/env python3
'''Function that calculates the integral'''


def poly_integral(poly, C=0):
    '''Function that calculates the integral of polynomial'''
    ls = []
    largo = len(poly)
    # Validate the list
    if type(poly) != list:
        return None
    # if not poly:
    #     return None
    if None in poly:
        return None
    # Validate C
    if not isinstance(C, (int, float)):
        return None
    if largo == 0:
        return ls.append(C)
    for i in range(largo):
        if isinstance(poly[i], (int, float)):
            if i == 0:
                ls.append(C)
                ls.append(poly[i])
            else:
                ls.append(poly[i] / (i + 1))
    for i in range(len(ls)):
        if ls[i] % 1 == 0:
            ls[i] = int(ls[i])
    if all(x == 0 for x in ls):
        return [0]
    return ls
