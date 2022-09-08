#!/usr/bin/env python3
'''
Modulus that creates a tensorflow layer that includes
L2 regularization
'''
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''


    Parameters
    ----------
    prev : TYPE tensor
        DESCRIPTION. Is a tensor containing the output of the
        previous layer
    n : TYPE int
        DESCRIPTION. Number of nodes
    activation : TYPE tensor
        DESCRIPTION. Type of activation to be used in the layer
    lambtha : TYPE float
        DESCRIPTION. L2 regularization parameter

    Returns
    -------
    The output of the new layer.

    '''
    init = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                 mode=("fan_avg"))
    l2 = tf.keras.regularizers.L2(lambtha)
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=l2
                            )(prev)
    return layer
