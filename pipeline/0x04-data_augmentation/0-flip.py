#!/usr/bin/env python3
"""0. Flip
"""
import tensorflow.compat.v1 as tf


def flip_image(image):
    """function that flips an image horizontally"""
    return tf.image.flip_left_right(image)
