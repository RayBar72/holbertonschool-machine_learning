#!/usr/bin/env python3
"""
4. Brightness
"""
import tensorflow.compat.v1 as tf


def change_brightness(image, max_delta):
    """function that randomly changes the brightness of an image"""
    return tf.image.random_brightness(image,
                                      max_delta)
