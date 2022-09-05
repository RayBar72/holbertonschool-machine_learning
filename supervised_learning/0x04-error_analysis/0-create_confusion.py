#!/usr/bin/env python3
'''
Module that creates a confusion matrix
'''
import numpy as np


def create_confusion_matrix(labels, logits):
    '''
    Creates a confusion matix

    Parameters
    ----------
    labels : TYPE numpy.ndarray
        DESCRIPTION.Contains the correct labels of each data points.
        is (m, classes). m is number of data points. classes is number of
        classes
    logits : TYPE numpy.ndarray
        DESCRIPTION. Contains predicted labels

    Returns
    -------
    A confusion numpy.ndarray of shape (classes, classes)
    with row indices representing the correct labels and column indices
    representing the predicted labels

    '''
    return np.matmul(labels.T, logits)
