#!/usr/bin/env python3
"""
3-q_learning
"""
import gym
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs Q-learning
    """
    initial_epsilon = epsilon
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        rewards = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, info = env.step(action)

            if done and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            state = new_state
            rewards += reward

            if done:
                break

        total_rewards.append(rewards)

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q, total_rewards
