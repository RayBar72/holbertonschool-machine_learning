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
    # if None in poly:
    #     return None
    # Validate C
    if not isinstance(C, (int, float)):
        return None
    if largo == 0:
        return ls.append(C)
    for i in range(largo):
        if isinstance(poly[i], (int, float)) and poly[i] != None:
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
    if len(ls) != 0:
        for i in range(len(ls) - 1, -1, - 1):
            if ls[i] == 0:
                print('valor i {}'.format(i))
                print('valro ls[i] {}'.format(ls[i]))
                ls.pop()
            else:
                break
    return ls
