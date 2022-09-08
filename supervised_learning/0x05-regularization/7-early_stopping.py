#!/usr/bin/env python3
'''
Modulus that determines if should be stopped gradient descent early
'''
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''
    Function that determines if you should stop gradient descent early

    Parameters
    ----------
    cost : TYPE float
        DESCRIPTION. Current validation cost of the NN
    opt_cost : TYPE float
        DESCRIPTION. Lowest recorded validation cost of NN
    threshold : TYPE
        DESCRIPTION. Threshold used for early stopping
    patience : TYPE int
        DESCRIPTION. Patience count used for early stopping
    count : TYPE int
        DESCRIPTION. Count of how long the threshold has not been met

    Returns
    -------
    A boolean of whether the network should be stopped early,
    followed by the updated count.

    '''
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0
    if count == patience:
        return (True, count)
    else:
        return (False, count)
