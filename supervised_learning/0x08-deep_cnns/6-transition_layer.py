#!/usr/bin/env python3
'''
Modulus that builds a transition layer as described in
"Densely Connected Convolutional Networks"
'''
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    '''
    Function that builds a transition layer

    Parameters
    ----------
    X : TYPE tensor
        DESCRIPTION. Output of the previus layer
    nb_filters : TYPE int
        DESCRIPTION. Number of filters in X
    compression : TYPE float
        DESCRIPTION. Compression factor for the transition layer

    Returns
    -------
    Transition layer and the number of filters.

    '''
    nb_filters = int(nb_filters * compression)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=nb_filters,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=2,
                                  padding='valid')(X)

    return X, nb_filters
