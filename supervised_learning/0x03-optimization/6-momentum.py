#!/usr/bin/env python3
'''Function that returns momentum optimaizer'''
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    '''Create momentum optimaizer'''
    momentum = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return momentum
