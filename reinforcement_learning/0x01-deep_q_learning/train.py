#!/usr/bin/env python3
"""train.py"""
import gym
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from keras import layers
import keras
from rl.processors import Processor
from PIL import Image
import numpy as np
from rl.callbacks import ModelIntervalCheckpoint, FileLogger


def create_q_model(actions=4, window=4):
    """Create the model for the agent"""
    inputs = layers.Input(shape=(window, 84, 84))

    x = layers.Permute((2, 3, 1))(inputs)

    x = layers.Conv2D(32, 8, strides=4, activation="relu",
                           data_format="channels_last")(x)
    x = layers.Conv2D(64, 4, strides=2, activation="relu",
                           data_format="channels_last")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu",
                           data_format="channels_last")(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation="relu")(x)

    action = layers.Dense(actions, activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=action)

class AtariProcessor(Processor):
    """Preprocess the images"""
    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """Rescale the images"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Clip the rewards between -1 and 1"""
        return np.clip(reward, -1., 1.)

def main():
    """ Main function """
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()
    model = create_q_model()
    model.summary()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                attr='eps',
                                value_max=1.,
                                value_min=.1,
                                value_test=.05,
                                nb_steps=1000000)
    memory = SequentialMemory(limit=1000000, window_length=4)
    agent = DQNAgent(model=model,
                    nb_actions=4,
                    policy=policy,
                    memory=memory,
                    processor=AtariProcessor(),
                    nb_steps_warmup=50000,
                    gamma=.99,
                    target_model_update=10000,
                    train_interval=4,
                    delta_clip=1.)
    agent.compile(keras.optimizers.Adam(lr=.00025), metrics=['mae'])
    agent.fit(env,
              nb_steps=50000,
              log_interval=10000,
              visualize=False,
              verbose=2)
    agent.save_weights('policy.h5', overwrite=True)


if __name__ == '__main__':
    main()
