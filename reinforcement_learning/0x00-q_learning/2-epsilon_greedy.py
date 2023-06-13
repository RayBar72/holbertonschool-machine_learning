#!/usr/bin/env python3
"""
2-epsilon_greedy
"""
import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action
    """
    p = np.random.uniform(0, 1)
    x = np.random.randint(Q.shape[1])
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = x
    return action
