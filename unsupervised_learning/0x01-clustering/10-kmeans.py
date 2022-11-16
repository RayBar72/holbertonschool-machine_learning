#!/usr/bin/env python3
'''
Modulus that performs K-means in a dataset
'''
import sklearn.cluster


def kmeans(X, k):
    '''
    Function that performs K-means in a dataset
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(k) is not int or k < 1:
        return None, None

    kmeans = sklearn.KMeans(k).fit(X)

    return kmeans.cluster_centers_, kmeans.labels_
