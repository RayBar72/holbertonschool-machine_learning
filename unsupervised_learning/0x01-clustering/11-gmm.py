#!/usr/bin/env python3
'''
Modulus that calculates a GMM from a dataset
'''
import sklearn.mixture


def gmm(X, k):
    '''
    Function that calculates a GMM from a dataset
    '''
    gm = sklearn.mixture.GaussianMixture(k).fit(X)

    pi = gm.weights_
    m = gm.means_
    S = gm.covariances_
    clss = gm.predict(X)
    bic = gm.bic(X)
    return pi, m, S, clss, bic
