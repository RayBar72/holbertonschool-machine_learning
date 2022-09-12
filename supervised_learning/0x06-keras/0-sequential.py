#!/usr/bin/env python3
'''
Modulus that creates a neural network with
keras library
'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''
    Function that builds a neural network with the Keras library

    Parameters
    ----------
    nx : TYPE int
        DESCRIPTION. Number of input features to the network
    layers : TYPE list
        DESCRIPTION. List containing the number of nodes in each
        layer of the network
    activations : TYPE list
        DESCRIPTION. List containing the activation functions used for
        each layer of the network
    lambtha : TYPE float
        DESCRIPTION. Is the L2 regularization parámeter
    keep_prob : TYPE float
        DESCRIPTION. probability that a node will be kept of dropout

    Returns
    -------
    The keras model.

    '''
    model = K.models.Sequential()
    la2 = K.regularizers.l2(lambtha)
    i = 0
    for layer, activa in zip(layers, activations):
        if i == 0:
            model.add(K.layers.Dense(layer,
                                     input_dim=nx,
                                     activation=activa,
                                     kernel_regularizer=la2))
            i += 1
        else:
            drop = K.layers.Dropout(rate=1 - keep_prob)
            model.add(drop)
            model.add(K.layers.Dense(layer,
                                     activation=activa,
                                     kernel_regularizer=la2))
    return model
