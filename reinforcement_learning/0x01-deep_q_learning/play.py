#!/usr/bin/env python3
""" play.py """

import gym
import keras
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

create_q_model = __import__('train').create_q_model
AtariProcessor = __import__('train').AtariProcessor


def main():
    env = gym.make('Breakout-v4')
    env.reset()

    model = create_q_model()
    memory = SequentialMemory(limit=1000000, window_length=4)

    dqn = DQNAgent(model=model,
                   nb_actions=4,
                   processor=AtariProcessor(),
                   memory=memory)

    dqn.compile(keras.optimizers.Adam(lr=.00025), metrics=['mae'])

    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=50, visualize=True)


if __name__ == '__main__':
    main()
