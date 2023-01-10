#!/usr/bin/env python3
"""
Modulus that creates a TF_IDF embedding
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Function that creates a TF_IDF embedding

    Parameters
    ----------
    sentences : List. Sentences to be analized
    vocab : List. vocabulary words to use in the analisys.
    The default is None.

    Returns
    -------
    embeddings. ndarray. Contains the embeddings
    features. List of the features used for embeddings

    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names_out()
    embeddings = x.toarray()
    return embeddings, features
