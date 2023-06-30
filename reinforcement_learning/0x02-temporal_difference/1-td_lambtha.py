#!/usr/bin/env python3
"""
Modulus for TD(λ)
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Function that performs the TD(λ) algorithm

    Arguments:
        env (gym): environment instance
        V (np.ndarray): value estimate
        policy (function): policy function for the agent
        lambtha (float): eligibility trace factor
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): learning rate
        gamma (float): discount rate

    Returns:
        (np.ndarray): the updated value estimate
    """

    for _ in range(episodes):
        S = env.reset()
        trace = np.zeros(env.observation_space.n)

        for _ in range(max_steps):
            A = policy(S)
            S_t1, R, done, _ = env.step(A)

            delta_t = R + gamma * V[S_t1] - V[S]
            trace = trace * lambtha * gamma
            trace[S] += 1.0

            V = V + alpha * delta_t * trace

            if done:
                break
            S = S_t1

    return V
