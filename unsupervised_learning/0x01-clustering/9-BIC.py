#!/usr/bin/env python3
'''
Modulus that finds the best number of cluster for
a GMM using Bayesian Information Criterion
'''
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    '''
    Function that that finds the best number of cluster for
    a GMM using Bayesian Information Criterion
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None, None)
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return (None, None, None, None)
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return (None, None, None, None)
    if kmin >= kmax:
        return (None, None, None, None)
    if type(iterations) != int or iterations <= 0:
        return (None, None, None, None)
    if type(tol) != float or tol <= 0:
        return (None, None, None, None)
    if type(verbose) != bool:
        return None, None, None, None

    best_k, best_results, l_totals, b_totals = [], [], [], []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        best_k.append(k)
        best_results.append((pi, m, S))
        l_totals.append(log_l)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        bic = p * np.log(n) - 2 * log_l
        b_totals.append(bic)
    b_totals = np.asarray(b_totals)
    best_b = np.argmin(b_totals)
    l_totals = np.asarray(l_totals)

    return best_k[best_b], best_results[best_b], l_totals, b_totals
