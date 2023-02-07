#!/usr/bin/env python3
"""
1-q_init
"""
import gym
import numpy as np


def q_init(env):
    """
    Function that initializes the Q-table
    """
    action = env.action_space.n
    state = env.observation_space.n
    q_table = np.zeros((state, action))
    return q_table
