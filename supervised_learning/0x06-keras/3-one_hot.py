#!/usr/bin/env python3
'''
Modulus that converts a label vector in to
one-hot matrix
'''
import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''
    Function that converts a label vector into a one-hot matrix

    Parameters
    ----------
    labels : TYPE vector
        DESCRIPTION. Vector to be converted in one-hot matrix
    classes : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None. The last dimension must be number of classes

    '''
    return K.utils.to_categorical(labels, num_classes=classes)
