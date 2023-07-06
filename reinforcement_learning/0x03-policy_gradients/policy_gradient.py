#!/usr/bin/env python3
"""
policy_gradient.py
"""
import numpy as np


def policy(matrix, weight):
    """Function that computes to policy with a weight of a matrix

    Args:
        matrix (np.array): Matrix
        weight (np.array): Weights

    Returns:
        np.array with the weight of a matrix
    """
    X = matrix @ weight
    Z = np.exp(X)  # type: ignore
    A = Z / np.sum(Z)
    return A


def policy_gradient(state, weight):
    """Function that computes the Monte-Carlo policy
    gradient based on a state and a weight matrix
    Args:
        state (np.array): Representation of the current state
        weight (np.array): Weights

    Returns:
        int, np.array: action and the gradient
    """
    P = policy(state, weight)

    a = np.random.choice(P.shape[1], p=P[0])

    delt = P.T
    delt = np.diagflat(delt) - (delt @ delt.T)
    delt = delt[a, :] / P[0, a]
    delt = state.T @ delt[np.newaxis, :]
    return a, delt
