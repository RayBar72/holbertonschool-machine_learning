#!/usr/bin/env python3
'''
Function that calculates the sumatory from:
    i = 1
    To n
    The function is i ** 2
'''


def sumation_recursive(ls, n):
    '''Function that calculates the sumatory recursive'''
    if n:
        i = n
        i = i ** 2
        ls.append(i)
        # print(ls)
        sumation_recursive(ls, n - 1)
    return ls


def summation_i_squared(n):
    '''Function that calculates sumatory'''
    if n < 1 or type(n) is not int or n is None:
        return None
    ls = []
    return sum(sumation_recursive(ls, n))
