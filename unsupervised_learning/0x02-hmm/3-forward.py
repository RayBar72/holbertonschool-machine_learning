#!/usr/bin/env python3
'''
Modulus that performs the forward algorithm for a HMM
'''
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    '''
    Function that performs the forward algorithm for a HMM
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
        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for n in range(N):
                F[n, t] = np.sum(F[:, t - 1] *
                                 Transition[:, n] *
                                 Emission[n, Observation[t]])

        P = F[:, -1].sum()

        return P, F

    except Exception:
        return None, None
