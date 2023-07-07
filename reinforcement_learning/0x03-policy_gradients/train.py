#!/usr/bin/env python3
"""
train.py
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.001, gamma=0.9, show_result=False):
    """Function that implements a full training

    Args:
        env (gym.enviroment): initial environment
        nb_episodes (int): number of episodes used for training
        alpha (float, optional): learning rate. Defaults to 0.000045.
        gamma (float, optional): discount rate. Defaults to 0.98.
    """
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    episode_rewards = []
    for episode in range(nb_episodes):
        state = env.reset()[np.newaxis, :]
        grads = []
        rewards = []
        score = 0
        T = 0
        done = False
        while not done:
            if show_result and not episode % 1000:
                env.render()
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[np.newaxis, :]
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state
            T += 1
            if done:
                break
        episode_rewards.append(score)
        print("Episode: " + str(episode) + " Score: " + str(score),
              end="\r", flush=False)
        for i in range(T):
            weight += alpha * grads[i] * sum([r * gamma ** r for t, r
                                              in enumerate(rewards[i:])])
    return episode_rewards
