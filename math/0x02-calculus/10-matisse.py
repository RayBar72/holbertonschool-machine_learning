#!/usr/bin/env python3
'''Function that casculates polynomial derivatives'''


def poly_derivative(poly):
    '''Function that casculates polynomial derivatives'''
    ls = []
    largo = len(poly)
    if type(poly) is not list:
        return None
    elif not poly:
        return None
    elif None in poly:
        return None
    elif largo == 0:
        return None
    elif largo == 1:
        return [0]
    else:
        for i in range(largo):
            if type(poly[i]) is int or type(poly[i]) is float:
                if i == 0:
                    pass
                else:
                    ls.append(poly[i] * i)
            else:
                return None
    return ls
