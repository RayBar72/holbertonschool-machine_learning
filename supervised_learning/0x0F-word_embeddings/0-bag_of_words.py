#!/usr/bin/env python3
"""
Modulus that creates a bag of words embedding matrix
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Function creates a bag of words embedding matrix

    Parameters
    ----------
    sentences : List. Sentences to analyze
    vocab : List of vocabulary words to be used in the analizis.
    The default is None.

    Returns
    -------
    embeddings. ndarray containnig embeddings
    features. List a fetures used for embeddings

    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names_out()
    embeddings = x.toarray()
    return embeddings, features
