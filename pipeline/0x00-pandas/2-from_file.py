#!/usr/bin/env python3
"""
2-from_file.py
"""
import numpy as np
import pandas as pd


def from_file(filename, delimiter):
    """Function that loads data from a file as a pd.DataFrame:

    Args:
        filename: File to load from.
        delimiter (str): column separetor
    """
    return pd.read_table(filename, sep=delimiter)
