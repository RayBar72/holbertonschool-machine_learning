#!/usr/bin/env python3
"""4-play"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Function that plays frozen lake game
    """
    state = env.reset()
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        env.render()
        if done:
            return reward
        state = new_state

    env.close()
