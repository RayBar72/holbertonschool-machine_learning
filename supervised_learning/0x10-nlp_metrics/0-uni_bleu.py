#!/usr/bin/env python3
"""
Modulus that calculates unigram BLEU score for a sentence
"""
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer


def dict_vecto(palabras, token_pattern='(?u)\\b\\w+\\b', ngram_range=(1, 1)):
    """_summary_

    Args:
        palabras (list): Function that creates a dictionary
        token_pattern (str, optional): Regrex for tokenization. Defaults to '(?u)\b\w+\b'.
        ngram_range (tuple, optional): ngram for tokens. Defaults to (1, 1).

    Returns:
        dict: Dictionary with the words and the number of times they appear
    """
    vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b',
                                 ngram_range=(1, 1))
    x = vectorizer.fit_transform(palabras)
    features = vectorizer.get_feature_names()
    embeddings = x.toarray().sum(axis=0)
    diccionario = dict(zip(features, embeddings))
    return diccionario


def references_dict(references, Sentence):
    """Function that creates a dictionary several times in fuction of the
    references list

    Args:
        references (list): References for the sentence
        Sentence (dict): dictionary with the words and
        the number of times they appear

    Returns:
        dict: with the words and the max number of times they appear
    """
    g_list = list(Sentence)
    g_list = {x: 0 for x in g_list}
    for ref in references:
        x = dict_vecto(ref)
        for k, v in g_list.items():
            if v < x.get(k, 0):
                g_list[k] = x[k]
    return g_list


def BP_function(sentence, references):
    """Function that calculates the brevity penalty"""
    c = len(sentence)
    r = len(references[np.argmin([abs(len(i) - c) for i in references])])
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r/c)
    return BP


def uni_bleu(references, sentence):
    """Function that calculates the unigram BLEU score for a sentence"""
    Sentence = dict_vecto(sentence)
    References = references_dict(references, Sentence)

    denominador = sum(list(Sentence.values()))
    numerador = sum(list(References.values()))
    total = numerador / denominador

    BP = BP_function(sentence, references)

    BLEU = BP * total

    return BLEU
