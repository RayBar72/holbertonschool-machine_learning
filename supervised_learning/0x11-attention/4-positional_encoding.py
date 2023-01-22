#!/usr/bin/env python3
"""
Modulus that calculates the positional encoding for a transformer
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer
    """
    t = np.arange(max_seq_len).reshape(max_seq_len, 1)
    indice = np.arange(dm).reshape(1, dm)
    dm_float = np.float32(dm)

    W = 1 / (np.power(10000, (2*(indice//2)/dm_float)))

    Wt = (W * t)

    positional = np.zeros((max_seq_len, dm))

    positional[:, 0::2] = np.sin(Wt[:, 0::2])

    positional[:, 1::2] = np.cos(Wt[:, 1::2])

    return positional
