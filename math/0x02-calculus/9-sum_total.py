#!/usr/bin/env python3
'''
Function that calculates the sumatory from:
    i = 1
    To n
    The function is i ** 2
'''

def rec_summation(n, end, steps, i =1, y = 1):
    if end == n:
        y += y
        print('y={}, i={}'.format(y, i))
        return y
    else:
        print('y={}, i={}'.format(y, i))
        rec_summation(n, end + steps, steps, i = i + steps, y = i ** 2)


def summation_i_squared(n):
    '''Function that calculates sumatory'''
    if type(n) is not int:
        return None
    if n <= 0:
        steps = -1
    else:
        steps = 1
    end = 0
    print('n = {}, end = {}, steps ={}'.format(n, end, steps))
    return rec_summation(n, end, steps)