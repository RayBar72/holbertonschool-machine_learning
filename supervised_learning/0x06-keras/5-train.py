#!/usr/bin/env python3
'''
Modulus that trins a model using mini-batch gradien descent
'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    '''
    Based on 4-train.py, update the function train_model
    to also analyze validaiton data

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
    validation_data : TYPE, optional
        DESCRIPTION. Dato to validate the model.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
    shuffle : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    if validation_data:
        validation_data = validation_data
    else:
        validation_data = None
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle)
