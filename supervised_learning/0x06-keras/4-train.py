#!/usr/bin/env python3
'''
Modulus that trins a model using mini-batch gradien descent
'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    '''
    Function that trains a model using mini-batch gradient descent

    Parameters
    ----------
    network : TYPE model
        DESCRIPTION. Model to be train
    data : TYPE numpy.ndarray
        DESCRIPTION. data is a numpy.ndarray of shape (m, nx) containing
        the input data
    labels : TYPE numpy.ndarray
        DESCRIPTION. (m, classes) containing the labels of data
    batch_size : TYPE int
        DESCRIPTION. Batch size used for mini-batch gradient descent
    epochs : TYPE int
        DESCRIPTION. Number of passes through data for mini-batch g.d.
    verbose : TYPE, optional
        DESCRIPTION. The default is True. Determines if output should be
        printed during the training
    shuffle : TYPE, optional
        DESCRIPTION. The default is False. Determines if shuffle the batches
        every epoch

    Returns
    -------
    History object generated after training model.

    '''
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle)
