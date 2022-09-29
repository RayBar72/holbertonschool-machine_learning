#!/usr/bin/env python3
'''
Modulus that builds a dense block as described in "Densely Connected
Convulutional Networks"
'''
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    '''
    Function that builds a dense block

    Parameters
    ----------
    X : TYPE tensor
        DESCRIPTION. Output from the previos layer
    nb_filters : TYPE integer
        DESCRIPTION. Represents the number of filters in X
    growth_rate : TYPE float
        DESCRIPTION. Is the number of layers in the dense block
    layers : TYPE
        DESCRIPTION.

    Returns
    -------
    Teh concatenated output and the number of filters within the
    the concatenated outputs.

    '''
    for i in range(layers):
        A = K.layers.BatchNormalization(axis=3)(X)
        A = K.layers.Activation('relu')(A)
        A = K.layers.Conv2D(growth_rate * 4,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer='he_normal')(A)

        A = K.layers.BatchNormalization(axis=3)(A)
        A = K.layers.Activation('relu')(A)
        A = K.layers.Conv2D(growth_rate,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer='he_normal')(A)

        A = K.layers.concatenate([X, A], axis=3)
        X = A
        nb_filters += growth_rate
    return X, nb_filters
