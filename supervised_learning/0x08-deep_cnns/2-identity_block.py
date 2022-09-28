#!/usr/bin/env python3
'''
Modulus that builds an identity block as described in Deep
Residual Learning for Image Recognition (2015)
'''
import tensorflow.keras as K


def identity_block(A_prev, filters):
    '''
    Function that builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015)

    Parameters
    ----------
    A_prev : TYPE numpy ndarray
        DESCRIPTION. Output of the previus layer
    filters : TYPE tuple
        DESCRIPTION. Conteins te different filters

    Returns
    -------
    The activated output of the identity block.

    '''
    F11, F3, F12 = filters
    C0 = K.layers.Conv2D(F11,
                         kernel_size=(1, 1),
                         padding='same',
                         # activation='relu',
                         kernel_initializer='he_normal')(A_prev)
    BN0 = K.layers.BatchNormalization(axis=3)(C0)
    R0 = K.layers.Activation('relu')(BN0)
    C1 = K.layers.Conv2D(F3,
                         kernel_size=(3, 3),
                         padding='same',
                         # activation='relu',
                         kernel_initializer='he_normal')(R0)
    BN1 = K.layers.BatchNormalization(axis=3)(C1)
    R1 = K.layers.Activation('relu')(BN1)
    C2 = K.layers.Conv2D(F12,
                         kernel_size=(1, 1),
                         padding='same',
                         # activation='relu',
                         kernel_initializer='he_normal')(R1)
    BN2 = K.layers.BatchNormalization(axis=3)(C2)
    AD = K.layers.Add()([BN2, A_prev])
    BL = K.layers.Activation('relu')(AD)
    return BL
