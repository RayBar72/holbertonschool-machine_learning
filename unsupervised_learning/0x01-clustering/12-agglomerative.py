#!/usr/env python3
'''
Modulus that performs agglomerative clustering
on dataset
'''
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    '''
    Function that performs agglomerative clustering
    on dataset
    '''
    # Cálculo de datos
    aglo = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(aglo, t=dist, criterion='distance')

    # Gráfica
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(aglo, color_threshold=dist)
    plt.show()
    return clss
