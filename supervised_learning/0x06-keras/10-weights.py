#!/usr/bin/env python3
'''
Modulus that has functions for:
    Save weights
    Load weights
'''
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    '''
    Function that saves a model’s weights

    Parameters
    ----------
    network : model
        DESCRIPTION. Model whose weights should be saved
    filename : TYPE str
        DESCRIPTION. Is the path of the file that the weights
        should be saved to
    save_format : TYPE, optional
        DESCRIPTION. The default is 'h5'.

    Returns
    -------
    None.

    '''
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    '''
    loads a model’s weights

    Parameters
    ----------
    network : TYPE model
        DESCRIPTION. Is the model to which the weights should be loaded
    filename : TYPE str
        DESCRIPTION. Is the path of the file that the weights should be
        loaded from

    Returns
    -------
    None.

    '''
    network.load_weights(filename)
