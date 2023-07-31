#!/usr/bin/env python3
"""
5. Hue
"""
import tensorflow.compat.v1 as tf


def change_hue(image, delta):
    """function that changes the hue of an image"""
    return tf.image.adjust_hue(image, delta)
