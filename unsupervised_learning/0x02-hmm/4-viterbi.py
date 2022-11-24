#!/usr/bin/env python3
'''
Modulus that calculates the must likely
sequence of hidden states for a HMM
'''
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    '''
    Function that calculates vitebi algorithm
    '''
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    T = Observation.shape[0]

    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if np.any(np.sum(Emission, axis=1) != 1):
        return None, None
    N, M = Emission.shape

    if type(Transition) is not np.ndarray or Transition.shape != (N, N):
        return None, None
    if np.any(np.sum(Transition, axis=1) != 1):
        return None, None

    if type(Initial) is not np.ndarray or Initial.shape != (N, 1):
        return None, None

    try:
        vito = np.zeros((N, T))

        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.amax(Transitions
                                  * F[:, t - 1]
                                  * Emissions)
                vito[n, t - 1] = np.argmax(Transitions *
                                           F[:, t - 1] *
                                           Emissions)

        path = [0 for i in range(T)]
        last_state = np.argmax(F[:, T - 1])
        path[0] = last_state

        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            path[backtrack_index] = int(vito[int(last_state), i])
            last_state = vito[int(last_state), i]
            backtrack_index += 1

        path.reverse()

        P = np.amax(F[:, T - 1], axis=0)

        return path, P

    except Exception:
        None, None
