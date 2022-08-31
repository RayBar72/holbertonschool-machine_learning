#!/usr/bin/env python3
'''Function that returns learning_rate_decay'''
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Function that returns learning_rate_decay'''
    alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return alpha
