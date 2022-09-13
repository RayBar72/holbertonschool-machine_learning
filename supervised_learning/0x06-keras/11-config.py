#!/usr/bin/env python3
'''
Modulus that has functions that saves and loads model configuration
'''
import tensorflow.keras as K


def save_config(network, filename):
    '''
    saves a model’s configuration in JSON format

    Parameters
    ----------
    network : TYPE model
        DESCRIPTION. Is the model whose configuration should be saved
    filename : TYPE str
        DESCRIPTION. Is the path of the file that the configuration
        should be saved to

    Returns
    -------
    None.

    '''
    mj = network.to_json()
    with open(filename, 'w') as f:
        f.write(mj)
    return None


def load_config(filename):
    '''
    loads a model with a specific configuration

    Parameters
    ----------
    filename : TYPE str
        DESCRIPTION.Is the path of the file containing the model’s
        configuration in JSON format

    Returns
    -------
    Loaded model.

    '''
    with open(filename, 'r') as f:
        loaded = f.read()
    model = K.models.model_from_json(loaded)
    return model
