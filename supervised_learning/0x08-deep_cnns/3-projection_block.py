#!/usr/bin/env python3
'''
Modulus that builds a projection block as described in Deep Residual
Learning for Image Recognition (2015)
'''
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    '''
    Function that builds a projection block as described in Deep Residual
    Learning for Image Recognition (2015)

    Parameters
    ----------
    A_prev : TYPE numpy ndarray
        DESCRIPTION. Output of the previus layer
    filters : TYPE tuple
        DESCRIPTION. Contains the number of filters
    s : TYPE, optional
        DESCRIPTION. Stride of the first convolutional in the
        main path and the shortcur. The default is 2.

    Returns
    -------
    The activated output of the projection block.

    '''
    F11, F3, F12 = filters
    C0A = K.layers.Conv2D(F11,
                          kernel_size=(1, 1),
                          # activation='relu',
                          strides=s,
                          padding='same',
                          kernel_initializer='he_normal')(A_prev)
    B0A = K.layers.BatchNormalization(axis=3)(C0A)
    R0 = K.layers.Activation('relu')(B0A)
    C1 = K.layers.Conv2D(F3,
                         kernel_size=(3, 3),
                         # activation='relu',
                         padding='same',
                         kernel_initializer='he_normal')(R0)
    B1 = K.layers.BatchNormalization(axis=3)(C1)
    R1 = K.layers.Activation('relu')(B1)
    C2 = K.layers.Conv2D(F12,
                         kernel_size=(1, 1),
                         # activation='relu',
                         padding='same',
                         kernel_initializer='he_normal')(R1)
    C0B = K.layers.Conv2D(F12,
                          kernel_size=(1, 1),
                          # activation='relu',
                          strides=s,
                          padding='same',
                          kernel_initializer='he_normal')(A_prev)
    B2 = K.layers.BatchNormalization(axis=3)(C2)
    B0B = K.layers.BatchNormalization(axis=3)(C0B)
    AD = K.layers.Add()([B2, B0B])
    Block = K.layers.Activation('relu')(AD)
    return Block
