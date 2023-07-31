#!/usr/bin/env python3
"""
1. Crop
"""
import tensorflow.compat.v1 as tf


def crop_image(image, size):
    """function that performs a random crop of an image"""
    return tf.image.random_crop(image, size)
