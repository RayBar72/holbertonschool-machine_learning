#!/usr/bin/env python3
"""
100-pca
"""
import tensorflow.compat.v1 as tf


def pca_color(image, alphas):
    """
    Function that performs PCA color augmentation
    as described in the AlexNet paper
    """
    # Convertir la imagen a float32
    image = tf.cast(image, tf.float32)

    # Redimensionar la imagen a un vector 2D
    image_reshaped = tf.reshape(image, (-1, 3))

    # Calcular la media
    mean = tf.reduce_mean(image_reshaped, axis=0)

    # Centrar los datos restando la media
    image_centered = image_reshaped - mean

    # Calcular la matriz de covarianza
    cov = tf.matmul(image_centered, image_centered, transpose_a=True)\
        / tf.cast(tf.shape(image_centered)[0], tf.float32)

    # Calcular los eigenvectores y eigenvalores de la matriz de covarianza
    e, v = tf.linalg.eigh(cov)

    # Crear una matriz con los alphas y los eigenvalores
    alpha_lambda = tf.math.multiply(alphas, tf.sqrt(e))

    # Calcular el cambio de color
    delta = tf.tensordot(v, alpha_lambda, axes=1)

    # Añadir el cambio de color a la imagen original
    image_augmented = image + delta

    # Asegurarse de que los valores de los píxeles estén en el rango correcto
    image_augmented = tf.cast(tf.clip_by_value(image_augmented, 0, 255),
                              dtype=tf.int32)

    return image_augmented
