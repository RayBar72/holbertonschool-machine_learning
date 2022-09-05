#!/usr/bin/env python3
'''
Modulus that calculates the specificity for each class
in a confusion matrix
'''
import numpy as np


def specificity(confusion):
    '''
    Function that calculates the specificity for each class in
    a confusion matrix

    Parameters
    ----------
    confusion : TYPE numpy.ndarray
        DESCRIPTION. Confusion is a confusion numpy.ndarray of shape
        (classes, classes) where row indices represent the correct
        labels and column indices represent the predicted labels

    Returns
    -------
    A numpy.ndarray of shape (classes,) containing the specificity
    of each class.

    '''
    t_p = np.diagonal(confusion, axis1=0, axis2=1)
    f_p = np.sum(confusion, axis=0) - t_p
    f_n = np.sum(confusion, axis=1) - t_p
    t_n = np.sum(confusion) - (t_p + f_p + f_n)
    speci = t_n / (t_n + f_p)
    return speci
