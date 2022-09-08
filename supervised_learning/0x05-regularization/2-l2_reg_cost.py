#!/usr/bin/env python3
'''
Moudulus that calculates the cost of a NN with l2 regularization
'''
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    '''
    Function that calculates the cost of a neural network
    with L2 regularization

    Parameters
    ----------
    cost : TYPE tensor
        DESCRIPTION. Tensor containing the cost
        of the network without L2 regularization

    Returns
    -------
    A tensor containing the cost of the network accounting for
    L2 regularization.

    '''
    return cost + tf.losses.get_regularization_losses()
