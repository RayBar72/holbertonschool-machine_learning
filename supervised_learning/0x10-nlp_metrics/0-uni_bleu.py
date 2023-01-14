#!/usr/bin/env python3
"""
Modulus that calculates unigram BLEU score for a sentence
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def dict_vecto(palabras, token_pattern='(?u)\\b\\w+\\b', ngram_range=(1, 1)):
    vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b',
                                 ngram_range=(1, 1))
    x = vectorizer.fit_transform(palabras)
    features = vectorizer.get_feature_names()
    embeddings = x.toarray().sum(axis=0)
    diccionario = dict(zip(features, embeddings))
    return diccionario


def references_dict(references, Sentence):
    g_list = list(Sentence)
    g_list = {x: 0 for x in g_list}
    for ref in references:
        x = dict_vecto(ref)
        for k, v in g_list.items():
            if v < x.get(k, 0):
                g_list[k] = x[k]
    return g_list


def BP_function(sentence, references):
    c = len(sentence)
    r = len(references[np.argmin([abs(len(i) - c) for i in references])])
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r/c)
    return BP


def uni_bleu(references, sentence):
    Sentence = dict_vecto(sentence)
    References = references_dict(references, Sentence)

    denominador = sum(list(Sentence.values()))
    numerador = sum(list(References.values()))
    total = numerador / denominador

    BP = BP_function(sentence, references)

    BLEU = BP * total

    return BLEU
