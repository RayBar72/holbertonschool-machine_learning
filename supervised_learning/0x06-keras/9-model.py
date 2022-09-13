#!/usr/bin/env python3
'''
Modulus that saves and loads a model
'''
import tensorflow.keras as K


def save_model(network, filename):
    '''
    saves an entire model

    Parameters
    ----------
    network : TYPE model
        DESCRIPTION. Model to be saved
    filename : TYPE string
        DESCRIPTION. Path to the file in which model should be saved

    Returns
    -------
    None.

    '''
    network.save(filename)


def load_model(filename):
    '''
    loads an entire model

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION. Path of the file that the model should be loaded from
        file is

    Returns
    -------
    The loaded model.

    '''
    return K.models.load_model(filename)
