#!/usr/bin/env python3
"""
1-from_dictionary.py
"""
import pandas as pd


diccionario = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}
frame = pd.DataFrame(diccionario, index=['A', 'B', 'C', 'D'])
globals()['df'] = frame
