#!/usr/bin/env python3
"""
Module for function SARSA(λ)
"""
import numpy as np


def epsilon_greedy(Q, S, epsilon):
    """
    Function that uses epsilon-greedy to determine the next A
    """
    p = np.random.uniform(0, 1)
    x = np.random.randint(Q.shape[1])
    if p > epsilon:
        A = np.argmax(Q[S, :])
    else:
        A = x
    return A


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """_summary_

    Args:
        env (gym): environment instance.
        Q (np.ndarray): Q table.
        lambtha (float): elegibility trace factor.
        episodes (int, optional): number of episodes. Defaults to 5000.
        max_steps (int, optional): max number of steps per episode.
            Defaults to 100.
        alpha (float, optional): learning rate. Defaults to 0.1.
        gamma (float, optional): descount rate. Defaults to 0.99.
        epsilon (int, optional): initial threshold for epsilon gready.
            Defaults to 1.
        min_epsilon (float, optional): min epsilon. Defaults to 0.1.
        epsilon_decay (float, optional): decay for epsilon. Defaults to 0.05.

    Returns:
        Q (np.ndarray): updated Q table.
    """
    for i in range(episodes):
        S = env.reset()
        E = np.zeros((Q.shape))
        A = epsilon_greedy(Q, S, epsilon=epsilon)
        for _ in range(max_steps):
            S_t1, reward, done, _ = env.step(A)
            A_t1 = epsilon_greedy(Q, S_t1, epsilon=epsilon)
            delta_t = reward + gamma * Q[S_t1, A_t1] - Q[S, A]
            E = E * lambtha * gamma
            E[S, A] += 1.0
            Q = Q + alpha * delta_t * E
            if done:
                break
            S = S_t1
            A = A_t1
        epsilon = min_epsilon + (epsilon - min_epsilon) *\
            np.exp(-(epsilon_decay * (i)))

    return Q
