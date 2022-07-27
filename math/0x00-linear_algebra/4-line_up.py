#!/usr/bin/env python3
'''Function that add two array'''


def add_arrays(arr1, arr2):
    '''Add two arrays'''
    if len(arr1) != len(arr2):
        return None
    list0 = []
    for x in range(len(arr1)):
        list0.append(arr1[x] + arr2[x])
    return list0
