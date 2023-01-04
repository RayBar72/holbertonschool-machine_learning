#!/usr/bin/env python3
""""
Modulus that performs several functions that:
    cleans data
    splits data in train, validation and testing sets
"""
import numpy as np
import pandas as pd


def cleans(cfile):
    """Function that cleans data

    Args:
        cfile (str): 
    """
    with open(cfile, 'r') as f:
        df = pd.read_csv(f, delimiter=',')
    # Timestamp to date format
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
    # Taking out NAs, duplicates and droping values before
    df = df.dropna()
    df = df.drop_duplicates(subset='Timestamp')
    df = df[df['Date'] >= '2017']
    # Sampling by hours
    df = df.set_index('Date')
    df = df.resample('H').agg({'Open': 'mean',
                            'High': 'mean',
                            "Low": 'mean',
                            'Close': 'mean',
                            'Volume_(BTC)': 'sum',
                            'Volume_(Currency)': 'sum',
                            'Weighted_Price': 'mean'})
    df.reset_index(inplace=True)
    # Dropping not relevant vars
    df.pop('Open')
    df.pop('High')
    df.pop('Low')
    df.pop('Volume_(BTC)')
    df.pop('Volume_(Currency)')
    df.pop('Weighted_Price')
    df = df.dropna()
    print(df.head())
    return df

def train_val_test(df):
    """Function that splits sets and standarize they

    Args:
        df (pd): pandas object
    """
    # Spliting sets
    date_time = pd.to_datetime(df.pop('Date'), format='%d.%m.%Y %H:%M:%S')
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    # Standarizing
    t_mean = train_df.mean()
    t_std = train_df.std()
    train_df = (train_df - t_mean) / t_std
    val_df = (val_df - t_mean) / t_std
    test_df = (test_df - t_mean) / t_std
    return train_df, val_df, test_df
