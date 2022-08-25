#!/usr/bin/env python3
'''
Modulus that calculates the softmax cross-entropy
loss of a prediction
'''
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    '''
    Calculates the softmax cross-entropy loss of a prediction

    Parameters
    ----------
    y : Placeholder
        Labels of input data.
    y_pred : Tensor
        Network predictions.

    Returns
    -------
    Tensor containing the loss of the prediction

    '''
    return tf.losses.softmax_cross_entropy(y, y_pred)
