#!/usr/bin/env python3
'''Function that casculates polynomial derivatives'''


def poly_derivative(poly):
    '''Function that casculates polynomial derivatives'''
    ls = []
    largo = len(poly)
    if type(poly) != list:
        return None
    elif not poly:
        return None
    elif largo == 0:
        return None
    elif largo == 1:
        return ls.append(0)
    else:
        for i in range(largo):
            if i == 0:
                pass
            else:
                ls.append(poly[i] * i)
    return ls
