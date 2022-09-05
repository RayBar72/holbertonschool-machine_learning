#!/usr/bin/env python3
'''
Modulus that calculates the sensitivity for each class in
confusion matrix
'''
import numpy as np


def sensitivity(confusion):
    '''
    

    Parameters
    ----------
    confusion : TYPE numpy.dnarray
        DESCRIPTION. confusion is a confusion numpy.ndarray of shape
        (classes, classes) where row indices represent the correct
        labels and column indices represent the predicted labels

    Returns
    -------
    A numpy.ndarray of shape (classes,) containing the sensitivity
    of each class

    '''
    diagonal = np.diagonal(confusion, axis1=0, axis2=1)
    total = np.sum(confusion, axis=1)
    retorno = np.divide(diagonal, total.T)
    return retorno
