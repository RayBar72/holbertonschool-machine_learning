#!/usr/bin/env python3
'''Function that creates_Adam_op'''
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''Function that creates_Adam_op'''
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
    return adam
