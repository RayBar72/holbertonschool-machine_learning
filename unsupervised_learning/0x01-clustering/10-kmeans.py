#!/usr/bin/env python3
'''
Modulus that performs K-means in a dataset
'''
import numpy as np
import sklearn.cluster as cl


def kmeans(X, k):
    '''
    Function that performs K-means in a dataset
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(k) is not int or k < 1:
        return None, None

    kmeans = cl.KMeans(k, random_state=0).fit(X)

    return kmeans.cluster_centers_, kmeans.labels_
