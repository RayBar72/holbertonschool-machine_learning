#!/usr/bin/env python3
"""
2-from_file.py
"""
import numpy as np
import pandas as pd

"""
Function that createes a pd.DataFrame from a dictionary

Returns
-------
pd.DataFrame.

"""
dictionary = {
    'index': ['A', 'B', 'C', 'D'],
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
    }
dataf = pd.DataFrame(dictionary)
globals()['pd'] = dataf
