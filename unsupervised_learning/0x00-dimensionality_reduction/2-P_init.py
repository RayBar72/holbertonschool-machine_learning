#!/usr/bin/env python3
'''Modulus that initializes variables for t-SNE'''
import numpy as np


def P_init(X, perplexity):
    '''Function that calculates the P affinities in t-SNE'''
    n, d = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0.0)
    P = np.zeros(shape=[n, n])
    betas = np.ones(shape=[n, 1])
    H = np.log2(perplexity)
    return D, P, betas, H
