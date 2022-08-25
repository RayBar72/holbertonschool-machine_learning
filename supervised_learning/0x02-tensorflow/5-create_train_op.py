#!/usr/bin/env python3
'''
Modulus that creates training operation for network
'''
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    '''
    Function that  creates the training operation for the network

    Parameters
    ----------
    loss : Tensor
        Loss of the network prediction.
    alpha : Float
        Learning rate.

    Returns
    -------
    Operation that trains network using gradien descent.

    '''
    train = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return train
