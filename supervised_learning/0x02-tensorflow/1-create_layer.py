#!/usr/bin/env python3
'''
Modulo that creates layers with tensor flow
'''
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    '''
    Function that creates layers for a NN

    Parameters
    ----------
    prev : np.narray
        Tensor output of the previous layer.
    n : int
        Number of nodes in the layer.
    activation : Function
        Activation function that the layer should use.

    Returns: the tensor output of layer
    -------
    None.

    '''
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=activa, name='layer')
    return layer(prev)
