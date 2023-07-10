#!/usr/bin/env python3
"""
1-from_dictionary.py
"""
import pandas as pd


diccionario = {
    'First': pd.Series([0.0, 0.5, 1.0, 1.5], index=['A', 'B', 'C', 'D']),
    'Second': pd.Series(['one', 'two', 'three', 'four'], index=['A', 'B', 'C', 'D'])
}
frame = pd.DataFrame(diccionario)
globals()['df'] = frame
