#!/usr/bin/env python3
"""Contains the gensim_to_keras function"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a keras Embedding layer
    :param model: trained gensim word2vec models
    :return: trainable keras Embedding
    """
    keys = model.wv
    weights = keys.vectors

    return tf.keras.layers.Embedding(input_dim=weights.shape[0],
                                     output_dim=weights.shape[1],
                                     weights=[weights])
