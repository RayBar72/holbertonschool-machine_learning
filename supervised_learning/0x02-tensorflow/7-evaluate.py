#!/usr/bin/env python3
'''
Modulus that evaluates the output of a neural network
'''
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    '''
    Function that evaluates the output of a neural network

    Parameters
    ----------
    X : numpy.ndarray
        Input data to evaluate.
    Y : numpy.ndarray
        One-hot labels for x.
    save_path : str
        saved model.

    Returns
    -------
    the network’s prediction, accuracy, and loss, respectively.

    '''
    with tf.Session() as session:
        saved = tf.train.import_meta_graph('{}.meta'.format(save_path))
        saved.restore(session, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        eva_y = session.run(y_pred, feed_dict={x: X, y: Y})
        eva_accuracy = session.run(accuracy, feed_dict={x: X, y: Y})
        eva_loss =session.run(loss, feed_dict={x: X, y: Y})
        return eva_y, eva_accuracy, eva_loss
