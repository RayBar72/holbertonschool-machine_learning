#!/usr/bin/env python3
'''
Modulus that contains a functiont that contains the
forward propagation graph for a neural network
'''
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    '''
    functiont that contains the forward propagation
    graph for a neural network

    Parameters
    ----------
    x : Placeholder
        Input data.
    layer_sizes : list
        DESCRIPTION. Contains the number of nodes in each layer.
    activations : list
        DESCRIPTION. Contains the activation functions for each layer.

    Returns: The prediction of network tensor form
    '''
    for i in range(len(layer_sizes)):
        if i == 0:
            estimation = create_layer(x, layer_sizes[i], activations[i])
        else:
            estimation = create_layer(estimation, layer_sizes[i],
                                      activations[i])
    return estimation
