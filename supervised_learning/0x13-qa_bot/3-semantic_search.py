#!/usr/bin/env python3
"""3. Semantic Search"""
from glob import glob
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Function that performs semantic search on a corpus of documents"""
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    documents = [sentence]
    for file in glob(corpus_path + '/*'):
        with open(file, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    outputs = embed(documents)
    corr = np.inner(outputs, outputs)
    index = np.argmax(corr[0, 1:])
    return documents[index + 1]
