#!/usr/bin/env python3
'''Function that casculates polynomial derivatives'''


def poly_derivative(poly):
    '''Function that casculates polynomial derivatives'''
    if type(poly) != list:
        return None
    if not poly:
        return None
    ls = []
    largo = len(poly)
    if largo == 0:
        return None
    if largo == 1:
        return ls.append(0)
    for i in range(largo):
        if i == 0:
            pass
        else:
            ls.append(poly[i] * i)
    return ls
