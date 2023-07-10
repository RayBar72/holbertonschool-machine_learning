#!/usr/bin/env python3
"""
0-from_numpy.py
"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray:

    Parameters
    ----------
    array : np.ndarray
        Is the np.ndarray from which you should create the pd.DataFrame.

    Returns
    -------
    pd.Dataframe.

    """
    alfabeto = list(map(chr, range(65, 65 + array.shape[1])))
    data = pd.DataFrame(array, columns=alfabeto)
    return data
