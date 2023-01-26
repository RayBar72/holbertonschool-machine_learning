#!/usr/bin/env python3
"""4-create_mask"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation
    """
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    largo = target.shape[1]
    a_mask = 1 - tf.linalg.band_part(tf.ones((largo, largo)), -1, 0)

    combined_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    combined_mask = combined_mask[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(combined_mask, a_mask)

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, combined_mask, decoder_mask
