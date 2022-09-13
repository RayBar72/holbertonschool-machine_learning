#!/usr/bin/env python3
'''
Modulus that contains function that tests a NN
'''
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''
    that tests a neural network

    Parameters
    ----------
    network : TYPE model
        DESCRIPTION.is the network model to test
    data : TYPE numpy.ndarray
        DESCRIPTION.is the input data to test the model with
    labels : TYPE one-hot
        DESCRIPTION. are the correct one-hot labels of data
    verbose : TYPE, optional
        DESCRIPTION. Is a boolean that determines if output should be
        printed during the testing process. The default is True.

    Returns
    -------
    The loss and accuracy of the model with the testing data, respectively.

    '''
    return network.evaluate(data, labels, verbose=verbose)
