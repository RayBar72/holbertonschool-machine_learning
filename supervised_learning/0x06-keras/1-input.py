#!/usr/bin/env python3
'''
Modulus that builds a NN with keares
Not using Sequential class
'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''
    Function that builds a neural network with the Keras library
    It is not allowed to use Sequential class

    Parameters
    ----------
    nx : TYPE. int
        DESCRIPTION. Number of input features
    layers : TYPE. list
        DESCRIPTION. List containing number of nodes in each layer
    activations : TYPE. list
        DESCRIPTION. List containing the activation function in each layer
    lambtha : TYPE. float
        DESCRIPTION. Regularization parameter
    keep_prob : TYPE. float
        DESCRIPTION. Probability that a node will be kept for dropout

    Returns
    -------
    Keras model.

    '''
    Inputs = K.Input(shape=(nx,))
    l2 = K.regularizers.l2(lambtha)
    i = 0
    for lay, act in zip(layers, activations):
        if i == 0:
            L = K.layers.Dense(lay,
                               activation=act,
                               kernel_regularizer=l2)(Inputs)
            i += 1
        else:
            L = K.layers.Dropout(1 - keep_prob)(L)
            L = K.layers.Dense(lay,
                               activation=act,
                               kernel_regularizer=l2)(L)
    model = K.Model(inputs=Inputs, outputs=L)
    return model
