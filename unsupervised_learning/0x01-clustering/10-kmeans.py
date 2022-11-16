#!/usr/bin/env python3
'''
Modulus that performs K-means in a dataset
'''
import sklearn.cluster


def kmeans(X, k):
    '''
    Function that performs K-means in a dataset
    '''
    kmeans = sklearn.cluster.KMeans(k).fit(X)

    return kmeans.cluster_centers_, kmeans.labels_
