#!/usr/bin/env python3
'''
Modulus that sets up Adam optimization for keras model
with categoriacl crossentropy loss and accuracy metrics
'''
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''
    Function that sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics

    Parameters
    ----------
    network : TYPE tensor
        DESCRIPTION. Model to optimize
    alpha : TYPE float
        DESCRIPTION. Learning rate
    beta1 : TYPE Float
        DESCRIPTION. First Adam optimization parameter
    beta2 : TYPE float
        DESCRIPTION. Second Adam optimization parameter

    Returns
    -------
    None.

    '''
    opt = K.optimizers.Adam(learning_rate=alpha,
                            beta_1=beta1,
                            beta_2=beta2)
    network.compile(optimizer=opt,
                    loss=K.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
    return None
