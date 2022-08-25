#!/usr/bin/env python3
'''
Modulos that contains a function that
returns two placeholders, x and y
'''
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    '''
    Function that returns two place holders, x an y

    Parameters
    ----------
    nx : int
        Number of feature columns in our data.
    classes : int
        Number of classes in our classifier.

    Returns
    -------
    Placeholders x and y.

    '''
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
