# 0x02. Hidden Markov Models #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is the Markov property?
- What is a Markov chain?
- What is a state?
- What is a transition probability/matrix?
- What is a stationary state?
- What is a regular Markov chain?
- How to determine if a transition matrix is regular
- What is an absorbing state?
- What is a transient state?
- What is a recurrent state?
- What is an absorbing Markov chain?
- What is a Hidden Markov Model?
- What is a hidden state?
- What is an observation?
- What is an emission probability/matrix?
- What is a Trellis diagram?
- What is the Forward algorithm and how do you implement it?
- What is decoding?
- What is the Viterbi algorithm and how do you implement it?
- What is the Forward-Backward algorithm and how do you implement it?
- What is the Baum-Welch algorithm and how do you implement it?

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Markov Chain | Write the function def markov_chain(P, s, t=1): that determines the probability of a markov chain being in a particular state after a specified number of iterations | 0-markov_chain.py |
| 1. Regular Chains | Write the function def regular(P): that determines the steady state probabilities of a regular markov chain | 1-regular.py |
| 2. Absorbing Chains | Write the function def absorbing(P): that determines if a markov chain is absorbing | 2-absorbing.py |
| 3. The Forward Algorithm | 3-forward.py |
| 4. The Viretbi Algorithm | Write the function def viterbi(Observation, Emission, Transition, Initial): that calculates the most likely sequence of hidden states for a hidden markov model | 4-viterbi.py |
| 5. The Backward Algorithm | Write the function def backward(Observation, Emission, Transition, Initial): that performs the backward algorithm for a hidden markov model | 5-backward.py |
| 6. The Baum-Welch Algorithm | Write the function def baum_welch(Observations, Transition, Emission, Initial, iterations=1000): that performs the Baum-Welch algorithm for a hidden markov model | 6-baum_welch.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
