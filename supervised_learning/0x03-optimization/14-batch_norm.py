#!/usr/bin/env python3
'''
Function that creates a batch normalization layer
for a neural network in tensorflow
'''
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    '''
    Function that creates a batch normalization layer
    for a neural network in tensorflow

    Parameters
    ----------
    prev : array
        Activated output of the previus layer.
    n : vector entero
        number of nodes in the layer to be created.
    activation : tensor
        Activation function.

    Returns
    -------
    Tensor
        Activated output of the layer.

    '''
    activa = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=None, kernel_initializer=activa, name='layer')
    Z = layer(prev)
    mu, sigma_2 = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]), name='gamma')
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name='beta')
    Z_b_norm = tf.nn.batch_normalization(
        Z, mu,
        sigma_2,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-8)
    return activation(Z_b_norm)
