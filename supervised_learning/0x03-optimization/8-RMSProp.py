#!/usr/bin/env python3
'''Function that returns RMSPropOptimizer'''
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''Function that returns RMSPropOptimizer'''
    rms = tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                    epsilon=epsilon).minimize(loss)
    return rms
