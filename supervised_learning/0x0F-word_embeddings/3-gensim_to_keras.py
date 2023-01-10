#!/usr/bin/env python3
"""
Modulus that converts a gensim word2vec to a keras
"""
from gensim.models import Word2Vec
import tensorflow.keras as K


def gensim_to_keras(model):
    """
    Function that converts a gensim word2vec to a keras

    Parameters
    ----------
    model : gensim model to be converted.

    Returns. Trainable keras embedding
    -------
    """
    return model.wv.get_keras_embedding(train_embeddings=False)
