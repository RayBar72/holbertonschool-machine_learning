#!/usr/bin/env python3
'''
Modulus that contains the function def calculate_accuracy(y, y_pred)
that calculates the accuracy of a prediction
'''
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    '''
    Function that calculates the accuracy of a prediction

    Parameters
    ----------
    y : Placeholder
        Labels of the input data.
    y_pred : Placeholder
        Networks predictions.

    Returns
    -------
    Tensor containing the decimal acuracy of the prediction.

    '''
    yes_not = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accura = tf.reduce_mean(tf.cast(yes_not, tf.float32))
    return accura
