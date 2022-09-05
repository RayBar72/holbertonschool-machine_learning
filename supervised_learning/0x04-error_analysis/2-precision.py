#!/usr/bin/env python3
'''
Modulas thatcalculates the precision for each class in a
confusion matrix
'''
import numpy as np


def precision(confusion):
    '''
    Calculates the precision for each class in a confusion matrix

    Parameters
    ----------
    confusion : TYPE numpy.ndarray
        DESCRIPTION. confusion is a confusion numpy.ndarray of shape
        (classes, classes) where row indices represent the correct
        labels and column indices represent the predicted labels

    Returns
    -------
    A numpy.ndarray of shape (classes,) containing the precision of
    each class.

    '''
    diagonal = np.diagonal(confusion, axis1=0, axis2=1)
    t_n = np.sum(confusion, axis=0).T
    retorno = np.divide(diagonal, t_n)
    return retorno
