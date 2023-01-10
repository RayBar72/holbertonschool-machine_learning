#!/usr/bin/env python3
"""
Modulus that creates and trains a genism fastText
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    Function that creates and trains a genism fastText
    """
    model = FastText(sentences=sentences,
                     vector_size=size,
                     min_count=min_count,
                     negative=negative,
                     window=window,
                     sg=cbow,
                     seed=seed,
                     workers=workers)
    model.build_vocab(sentences)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=iterations)
    return model
