#!/usr/bin/env python3
'''Moudulus that creates a layer of NN using droput'''
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''
    Function that creates a layer of a neural network using dropout

    Parameters
    ----------
    prev : TYPE tensor
        DESCRIPTION. Output of the previus layer
    n : TYPE int
        DESCRIPTION. Number of nodes
    activation : TYPE str
        DESCRIPTION. Activation functions that should be used in the layer
    keep_prob : TYPE float
        DESCRIPTION. Probability of the node to be kept

    Returns
    -------
    The output of the new layer.

    '''
    init = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                 mode=("fan_avg"))
    drop = tf.layers.Dropout(rate=keep_prob)
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=drop)(prev)
    return layer
