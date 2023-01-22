#!/usr/bin/env python3
"""
Modulus that calculates the scaled dot product attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Function that calculates the scaled dot product attention
    """
    uno = tf.matmul(Q, K, transpose_b=True)
    k = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = uno / tf.math.sqrt(k)

    if mask is not None:
        scaled += (mask * -1e9)

    weights = tf.nn.softmax(scaled, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
