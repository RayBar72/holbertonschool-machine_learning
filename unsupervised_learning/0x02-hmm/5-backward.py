#!/usr/bin/env python3
'''
Modulus that performs the backward algorithm
'''
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    '''
    Function that performs the backward algorithm
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
        beta = np.zeros([N, T])
        beta[:, -1] = np.ones([N])

        for t in range(T - 2, -1, -1):
            for n in range(N):
                Transitions = Transition[n, :]
                Emissions = Emission[:, Observation[t + 1]]
                beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)

        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
        return P, beta

    except Exception:
        return None, None
