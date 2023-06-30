#!/usr/bin/env python3
"""
Module for function Montecarlo
"""
import numpy as np


def game(env, S, policy, max_steps):
    """_summary_

    Args:
        env (gym): AI gym environment.
        S (gym): initial state of the environment.
        policy (function): takes in state and returns action.
        max_steps (int): max number of steps per episode.
    """
    lista = []
    for _ in range(max_steps):
        A = policy(S)
        S_t1, R, done, _ = env.step(A)
        lista.append((S, A, R))
        if done:
            break
        S = S_t1
    return np.array(lista, dtype=int)


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """_summary_

    Args:
        env (gym): AI gym environment.
        V (np.ndarray): value estimate.
        policy (function): takes in state and returns action.
        episodes (int, optional): total number of episodes. Defaults to 5000.
        max_steps (int, optional): max number of steps per
                                episode. Defaults to 100.
        alpha (float, optional): learning rate. Defaults to 0.1.
        gamma (float, optional): discount rate. Defaults to 0.99.

    Returns:
        (np.ndarray): updated value estimate.
    """
    for i in range(episodes):
        S = env.reset()

        G = 0
        episode = game(env, S, policy, max_steps)

        for steps in episode[::-1]:
            S, _, R = steps
            G = gamma * G + R
            if not (S in episode[:i, 0]):
                V[S] = V[S] + alpha * (G - V[S])

    return V
