#!/usr/bin/env python3
"""
Modulus that calculates unigram BLEU score for a sentence
"""
import numpy as np


def dict_vecto(palabras, ngram_range=1):
    """_summary_
    Args:
        palabras (list): Function that creates a dictionary
        ngram_range (int, optional): ngram for tokens. Defaults to 1.

    Returns:
        dict: Dictionary with the words and the number of times they appear
    """
    ngram_range = 1

    largo = len(palabras)

    posibles = []
    x = ""
    for i in range(0, largo):
        try:
            for j in range(ngram_range):
                x += palabras[i + j]
                if j != ngram_range - 1:
                    x += ' '
            posibles.append(x)
        except Exception as e:
            pass
        x = ""
    dicc = {x: posibles.count(x) for x in posibles}
    return dicc


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
