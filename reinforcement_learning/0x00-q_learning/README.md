# 0x00. Q-learning #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is a Markov Decision Process?
- What is an environment?
- What is an agent?
- What is a state?
- What is a policy function?
- What is a value function? a state-value function? an action-value function?
- What is a discount factor?
- What is the Bellman equation?
- What is epsilon greedy?
- What is Q-learning?

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Load the Environment | Write a function def load_frozen_lake(desc=None, map_name=None, is_slippery=False): that loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym | 0-load_env.py |
| 1. Initialize Q-table | Write a function def q_init(env): that initializes the Q-table | 1-q_init.py |
| 2. Epsilon Greedy | Write a function def epsilon_greedy(Q, state, epsilon): that uses epsilon-greedy to determine the next action | 2-epsilon_greedy.py |
| 3. Q-learning | Write the function def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05): that performs Q-learning | 3-q_learning.py |
| 4. Play | Write a function def play(env, Q, max_steps=100): that has the trained agent play an episode | 4-play.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
