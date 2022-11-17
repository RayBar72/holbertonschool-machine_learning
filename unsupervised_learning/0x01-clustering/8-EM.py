#!/usr/bin/env python3
'''
Modulus that performs the EM for GMM
'''
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    '''
    Function that performs the EM for GMM
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None

    if type(k) is not int or k < 1 or X.shape[0] < k:
        return None, None, None, None, None

    if type(iterations) is not int or iterations < 1:
        return None, None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    l0 = 0
    j = 0

    for i in range(iterations):
        g, l1 = expectation(X, pi, m, S)
        if i % 10 == 0 and verbose is True:
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(l1, 5)))
        if abs(l0 - l1) <= tol:
            if verbose is True:
                print('Log Likelihood after {} iterations: {}'.format(
                    i, round(l1, 5)))
            return pi, m, S, g, l1
        pi, m, S = maximization(X, g)
        l0 = l1
        j += 1

    g, l0 = expectation(X, pi, m, S)
    if verbose is True:
        print('Log Likelihood after {} iterations: {}'.format(
            j, round(l0, 5)))

    return pi, m, S, g, l0
