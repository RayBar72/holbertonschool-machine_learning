#!/usr/bin/env python3
'''
Modulus with function that makes prediction using a NN
'''
import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''
    that makes a prediction using a neural network

    Parameters
    ----------
    network : TYPE model
        DESCRIPTION. Is the network model to make the prediction with
    data : TYPE numpy.ndarray
        DESCRIPTION. Is the input data to make the prediction with
    verbose : TYPE, optional
        DESCRIPTION. is a boolean that determines if output should be
        printed during the prediction process. The default is False.

    Returns
    -------
    The prediction for the data.

    '''
    return network.predict(data, verbose=verbose)
