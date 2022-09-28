#!/usr/bin/env python3
'''
Modulus that builds an inception block as described in Going Deeper
with Convolutions (2014)
'''
import tensorflow.keras as K


def inception_block(A_prev, filters):
    '''
    Function that builds an inception block as described in Going Deeper
    with Convolutions (2014)

    Parameters
    ----------
    A_prev : TYPE tensor
        DESCRIPTION. Output of previous layer
    filters : TYPE tuple
        DESCRIPTION. Contains the filters of convolutional layers

    Returns
    -------
    Concatenated output.

    '''
    F1, F3R, F3, F5R, F5, FPP = filters
    C1 = K.layers.Conv2D(F1,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='relu')(A_prev)
    C2 = K.layers.Conv2D(F3R,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='relu')(A_prev)
    C3 = K.layers.Conv2D(F5R, kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer='he_normal',
                         activation='relu')(A_prev)
    P1 = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(A_prev)
    CC22 = K.layers.Conv2D(F3,
                           kernel_size=(3, 3),
                           padding='same',
                           kernel_initializer='he_normal',
                           activation='relu')(C2)
    CC32 = K.layers.Conv2D(F5,
                           kernel_size=(5, 5),
                           padding='same',
                           kernel_initializer='he_normal',
                           activation='relu')(C3)
    PC12 = K.layers.Conv2D(FPP,
                           kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer='he_normal',
                           activation='relu')(P1)
    concha = K.layers.Concatenate(axis=3)([C1,
                                           CC22,
                                           CC32,
                                           PC12])
    return concha
