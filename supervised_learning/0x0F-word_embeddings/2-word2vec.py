#!/usr/bin/env python3
"""
Modulus that creates and trains a gesim word2vec model
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Function that creates and traina gesim word2vec model
    """
    model = Word2Vec(sentences=sentences,
                     vector_size=size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=cbow,
                     seed=0,
                     workers=1)
    model.build_vocab(sentences)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=iterations)
    return model
