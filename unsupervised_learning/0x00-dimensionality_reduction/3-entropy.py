#!/usr/bin/env python3
'''Modulus that calculates the Shannon entropy and P affinities'''
import numpy as np


def HP(Di, beta):
    '''Function that calculates deh shannon entropy an P affinities'''
    P = np.exp(-Di * beta)
    sumP = np.sum(P)
    Pi = P / sumP
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
